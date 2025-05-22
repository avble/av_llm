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
#include "index.html.gz.hpp"

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

    std::string model_path;

    { // parsing the argument 
        int i = 0;
        for (int i = 1; i < argc; i++){
            if (strcmp(argv[i], "-m") == 0){
                if (i + 1 < argc){
                    model_path = argv[++i];
                }else {
                    print_usage(1, argv);
                    return 1;
                }
            }
        }
    }

    if (model_path == "")
    {
        print_usage(1, argv);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();

    llama_model *model = llama_load_model_from_file(model_path.c_str(), model_params);

    if (model == nullptr)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // print the token as text to screen
    static auto model_print_token = [&](llama_model *model, std::vector<llama_token> &tokens)
    {
        for (auto token_ : tokens)
        {
            char buf[128];
            int n = llama_token_to_piece(vocab, token_, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);
        }
    };

    struct avllm_context_param
    {
        llama_context *ctx;
        llama_sampler *smpl;
    };

    static auto model_context_init = [](llama_model *model, std::string &prompt) -> avllm_context_param
    {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        int n_predict = 32;

        const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.no_perf = false;
        ctx_params.n_ctx = n_prompt + n_predict - 1;
        ctx_params.n_batch = n_prompt;
    
        // allocate space for the tokens and tokenize the prompt
        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
            return {.ctx = nullptr, .smpl = nullptr};
        }
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        llama_context *ctx = llama_init_from_model(model, ctx_params);
        {   // context initialization
            if (ctx == NULL)
            {
                fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
                return {.ctx = nullptr, .smpl = nullptr};
            }
        }

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return {.ctx = nullptr, .smpl = nullptr};
        }

        // initialize the sampler
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler *smpl = llama_sampler_chain_init(sparams);

        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        return {.ctx = ctx, .smpl = smpl};
    };

    static auto model_context_deinit = [](avllm_context_param &ctx)
    {
        fprintf(stderr, "\n");
        llama_perf_sampler_print(ctx.smpl);
        llama_perf_context_print(ctx.ctx);
        fprintf(stderr, "\n");
        llama_sampler_free(ctx.smpl);
        llama_free(ctx.ctx);
        ctx.smpl = nullptr;
        ctx.ctx = nullptr;
    };

    static auto model_context_get_text = [&](llama_model *model, const avllm_context_param &ctx, std::function<int(int rc, const std::string &text)> func_hdl, int max_token = 1024)
    {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const auto t_main_start = ggml_time_us();
        llama_token new_token_id;
        int num_token = 0;
        while (num_token++ < max_token)
        {
            new_token_id = llama_sampler_sample(ctx.smpl, ctx.ctx, -1);

            // is it an end of generation?
            if (llama_token_is_eog(vocab, new_token_id))
            {
                func_hdl(-1, "");
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                break;
            }
            // std::string text(buf, n);
            // printf("%s", text.c_str());
            // fflush(stdout);

            llama_batch batch = llama_batch_get_one(&new_token_id, 1);
            // evaluate the current batch with the transformer model
            if (llama_decode(ctx.ctx, batch))
            {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                func_hdl(-1, "");
                break;
            }

            if (func_hdl(0, std::string(buf, n)) != 0)
                break;
        }
    };

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

        avllm_context_param avllm_ctx_ = model_context_init(model, prompt);
        if (avllm_ctx_.ctx == nullptr or avllm_ctx_.smpl == nullptr)
        {
            res.result() = http::status_code::internal_error;
            res.end();
        }

        res.event_source_start();

        auto get_text_hdl = [&](int rc, const std::string &text) -> int
        {
            if (rc == 0)
                res.chunk_write("data: " + oai_make_stream(text));

            return 0;
        };

        model_context_get_text(model, avllm_ctx_, get_text_hdl);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        res.chunk_end();
        model_context_deinit(avllm_ctx_);
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
    // route_.get("/", handle_static_file("simplechat/index.html", "text/html; charset=utf-8"));
    // route_.get("/index.html", handle_static_file("simplechat/index.html", "text/html; charset=utf-8"));
    // route_.get("/datautils.mjs", handle_static_file("simplechat/datautils.mjs", "text/javascript; charset=utf-8"));
    // route_.get("/simplechat.css", handle_static_file("simplechat/simplechat.css", "text/css; charset=utf-8"));
    // route_.get("/simplechat.js", handle_static_file("simplechat/simplechat.js", "text/javascript"));
    // route_.get("/ui.mjs", handle_static_file("simplechat/ui.mjs", "text/javascript; charset=utf-8"));


    static auto web_handler = [&](http::response res) -> void
    {

        if (res.reqwest().get_header("accept-encoding").find("gzip") == std::string::npos) {
            res.set_content("Error: gzip is not supported by this browser", "text/plain");
        } else {
            res.set_header("Content-Encoding", "gzip");
            // COEP and COOP headers, required by pyodide (python interpreter)
            res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
            res.set_header("Cross-Origin-Opener-Policy", "same-origin");
            res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
            // res.set_content(reinterpret_cast<std::>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
        }
        res.end();
    };
    route_.get("/", web_handler);
    route_.get("/index.html", web_handler);

    // OpenAI API
    // model
    route_.get("/models", handle_models);
    route_.get("/v1/models", handle_models);
    route_.get("/v1/models/{model_name}", [](http::response res) {});
    route_.post("/completions", [](http::response res)
                { std::thread{completions_chat_handler, std::move(res)}.detach(); });
    route_.post("/v1/completions", [](http::response res)
                { std::thread{completions_chat_handler, std::move(res)}.detach(); });
    // chat
    route_.post("/chat/completions", [](http::response res)
                { std::thread{completions_chat_handler, std::move(res)}.detach(); });
    route_.post("/v1/chat/completions",
                [](http::response res)
                { std::thread{completions_chat_handler, std::move(res)}.detach(); });

    http::start_server(8080, route_);

    LOG("\n");

    llama_free_model(model);
    return 0;
}
