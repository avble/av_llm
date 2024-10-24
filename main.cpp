/* Copyright (c) 2024-2024 Harry Le (avble.harry at gmail dot com)

It can be used, modified.
*/

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include "av_connect.hpp"
#include "helper.hpp"
#include "log.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#include <inttypes.h>

using json = nlohmann::ordered_json;
#define MIMETYPE_JSON "application/json; charset=utf-8"

static void print_usage(int, char **argv)
{
    AVLLM_LOG("\nexample usage:\n");
    AVLLM_LOG("\n    %s -m model.gguf\n", argv[0]);
    AVLLM_LOG("\n");
}

int main(int argc, char **argv)
{

    gpt_params params;

    params.prompt = "hello";
    params.n_predict = 1024;

    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, print_usage))
    {
        return 1;
    }

    gpt_init();

    // total length of the sequence including the prompt
    const int n_predict = params.n_predict;

    // init LLM
    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model
    llama_model_params model_params = llama_model_params_from_gpt_params(params);
    llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    static auto completions_chat_handler = [&](http::response res) -> void
    {
        AVLLM_LOG_TRACE_SCOPE("completions_chat_handler")

        json body_ = json::parse(res.reqwest().body());

        nlohmann::json messages_js = body_.at("messages");
        std::string prompt;

        // extract promt
        for (const auto &msg : messages_js)
        {
            std::string role = (msg.contains("role") and msg.at("role").is_string()) ? msg.at("role") : "";
            if (role == "user")
            {
                std::string content = msg.at("content").get<std::string>();
                prompt = content;
            }
        }

        llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
        llama_context *ctx = llama_new_context_with_model(model, ctx_params);

        if (ctx == NULL)
        {
            fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
            res.result() = http::status_code::internal_error;
            res.end();
            return;
        }

        // tokenize the prompt
        params.prompt = prompt;

        std::vector<llama_token> tokens_list;
        tokens_list = ::llama_tokenize(ctx, params.prompt, true);

        const int n_ctx = llama_n_ctx(ctx);
        const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size());

        AVLLM_LOG("\n");
        AVLLM_LOG_INFO("%s: n_predict = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_predict, n_ctx, n_kv_req);

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if (n_kv_req > n_ctx)
        {
            AVLLM_LOG_ERROR("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
            AVLLM_LOG_ERROR("%s:        either reduce n_predict or increase n_ctx\n", __func__);
        }

        // print the prompt token-by-token
        AVLLM_LOG("\n");

        for (auto id : tokens_list)
            AVLLM_LOG("%s", llama_token_to_piece(ctx, id).c_str());

        // create a llama_batch with size 512
        // we use this object to submit token data for decoding
        llama_batch batch = llama_batch_init(512, 0, 1);

        // evaluate the initial prompt
        for (size_t i = 0; i < tokens_list.size(); i++)
            llama_batch_add(batch, tokens_list[i], i, {0}, false);

        // llama_decode will output logits only for the last token of the prompt
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0)
            AVLLM_LOG("%s: llama_decode() failed\n", __func__);

        // main loop
        int n_cur = batch.n_tokens;

        res.set_header("Access-Control-Allow-Origin", res.reqwest().get_header("origin"));
        res.event_source_start();

        while (n_cur <= n_predict)
        {
            const llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict)
            {
                AVLLM_LOG("\n");
                break;
            }

            std::string token_str = llama_token_to_piece(ctx, new_token_id).c_str();
            res.chunk_write("data: " + oai_make_stream(token_str));
            AVLLM_LOG_DEBUG("[Chunk]: %s\n", token_str);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, {0}, true);

            n_cur += 1;

            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch))
            {
                AVLLM_LOG_ERROR("%s : failed to eval, return code %d\n", __func__, 1);
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        res.chunk_end();

        llama_free(ctx);
    };

    static auto handle_models = [&](http::response res)
    {
        AVLLM_LOG_TRACE_SCOPE("handle_models")
        json models = {{"object", "list"},
                       {"data",
                        {
                            {{"id", "openchat_3.6"},
                             {"object", "model"},
                             {"created", std::time(0)},
                             {"owned_by", "avbl llm"},
                             {"meta", "meta"}},
                        }}};

        res.set_content(models.dump(), MIMETYPE_JSON);
        res.end();
    };

    static auto completions_handler = [&](http::response res) -> void
    {
        logger_function_trace_llamacpp __trace("", "completions_handler");

        AVLLM_LOG_DEBUG("completions_handler received body: %s\n", res.reqwest().body().c_str());

        json body_ = json::parse(res.reqwest().body());

        // check if it is stream
        bool is_stream = false;
        if (body_.contains("stream"))
        {
            json js_stream = body_.at("stream");
            if (not(js_stream.is_boolean() and js_stream.get<bool>() == true))
            {
                res.result() = http::status_code::internal_error;
                res.end();
            }
            is_stream = true;
        }

        std::string prompt;
        if (not(body_.contains("prompt") and body_.at("prompt").is_string()))
        {
            res.result() = http::status_code::internal_error;
            res.end();
        }
        prompt = body_.at("prompt").get<std::string>();
        AVLLM_LOG_DEBUG("prompt: %s\n", prompt.c_str());

        llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
        llama_context *ctx = llama_new_context_with_model(model, ctx_params);

        if (ctx == NULL)
        {
            fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
            res.result() = http::status_code::internal_error;
            return;
        }

        // tokenize the prompt
        params.prompt = prompt;

        std::vector<llama_token> tokens_list;
        tokens_list = ::llama_tokenize(ctx, params.prompt, true);

        const int n_ctx = llama_n_ctx(ctx);
        const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size());

        AVLLM_LOG("\n");
        AVLLM_LOG_INFO("%s: n_predict = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_predict, n_ctx, n_kv_req);

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if (n_kv_req > n_ctx)
        {
            AVLLM_LOG_ERROR("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
            AVLLM_LOG_ERROR("%s:        either reduce n_predict or increase n_ctx\n", __func__);
        }

        // print the prompt token-by-token
        AVLLM_LOG("\n");

        for (auto id : tokens_list)
            AVLLM_LOG("%s", llama_token_to_piece(ctx, id).c_str());

        // create a llama_batch with size 512
        // we use this object to submit token data for decoding
        llama_batch batch = llama_batch_init(512, 0, 1);

        // evaluate the initial prompt
        for (size_t i = 0; i < tokens_list.size(); i++)
            llama_batch_add(batch, tokens_list[i], i, {0}, false);

        // llama_decode will output logits only for the last token of the prompt
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0)
        {
            AVLLM_LOG("%s: llama_decode() failed\n", __func__);
            llama_free(ctx);
            res.result() = http::status_code::internal_error;
            res.end();
            return;
        }

        // main loop
        int n_cur = batch.n_tokens;

        res.set_header("Access-Control-Allow-Origin", res.reqwest().get_header("origin"));
        res.event_source_start();

        while (n_cur <= n_predict)
        {
            const llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict)
            {
                AVLLM_LOG("\n");
                break;
            }

            std::string str_token_id = llama_token_to_piece(ctx, new_token_id);
            res.chunk_write("data: " + oai_make_stream(str_token_id, false));

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, {0}, true);

            n_cur += 1;

            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch))
            {
                AVLLM_LOG_ERROR("%s : failed to eval, return code %d\n", __func__, 1);
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        res.chunk_end();
        llama_free(ctx);
    };

    struct handle_static_file
    {
        handle_static_file(std::string _file_path, std::string _content_type)
        {
            file_path = _file_path;
            content_type = _content_type;
        }
        void operator()(http::response res)
        {
            AVLLM_LOG_TRACE_SCOPE("handle_static_file")
            if (not std::filesystem::exists(std::filesystem::path(file_path)))
            {
                res.result() = http::status_code::not_found;
                res.end();
                return;
            }

            res.set_header("Content-Type", content_type);

            std::ifstream infile(file_path);
            std::stringstream str_stream;
            str_stream << infile.rdbuf();
            res.set_content(str_stream.str(), content_type);
            res.end();
        }

        std::string file_path;
        std::string content_type;
    };

    static auto preflight = [](http::response res)
    {
        res.set_header("Access-Control-Allow-Origin", res.reqwest().get_header("origin"));
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_content("", "text/html");
        res.end();
    };

    http::route route_;

    route_.set_option_handler(preflight);

    // Web (yeap, I known it currently it reads from file each time!)
    route_.get("/", handle_static_file("simplechat/index.html", "text/html; charset=utf-8"));
    route_.get("/index.html", handle_static_file("simplechat/index.html", "text/html; charset=utf-8"));
    route_.get("/datautils.mjs", handle_static_file("simplechat/datautils.mjs", "text/javascript; charset=utf-8"));
    route_.get("/simplechat.css", handle_static_file("simplechat/simplechat.css", "text/css; charset=utf-8"));
    route_.get("/simplechat.js", handle_static_file("simplechat/simplechat.js", "text/javascript"));
    route_.get("/ui.mjs", handle_static_file("simplechat/ui.mjs", "text/javascript; charset=utf-8"));

    // OpenAI API
    // model
    route_.get("/models", handle_models);
    route_.get("/v1/models", handle_models);
    route_.get("/v1/models/{model_name}", [](http::response res) {});
    // completions: legacy
    route_.post("/completions", [](http::response res)
                { std::thread{completions_handler, std::move(res)}.detach(); });
    route_.post("/v1/completions", [](http::response res)
                { std::thread{completions_handler, std::move(res)}.detach(); });
    // chat
    route_.post("/chat/completions", [](http::response res)
                { std::thread{completions_chat_handler, std::move(res)}.detach(); });
    route_.post("/v1/chat/completions",
                [](http::response res)
                { std::thread{completions_chat_handler, std::move(res)}.detach(); });

    http::start_server(8080, route_);

    LOG("\n");

    llama_sampler_free(smpl);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
