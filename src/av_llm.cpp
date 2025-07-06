/* Copyright (c) 2024-2024 Harry Le (avble.harry at gmail dot com)

It can be used, modified.
*/

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama-cpp.h"
#include "llama.h"
#include "log.h"
#include "log.hpp"
#include "sampling.h"

#include "av_connect.hpp"
#include "helper.hpp"
#include "model.hpp"
#include "openai.hpp"
namespace av_llm {
#include "index.html.gz.hpp"
}

#include "utils.hpp"

#include <CLI/CLI.hpp>
#include <inttypes.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <dbghelp.h>
#include <windows.h>
#pragma comment(lib, "dbghelp.lib")
#else
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#endif

#ifdef _MSC_VER
#include <ciso646>
#endif

// prototype
static void model_cmd_handler(std::string sub_cmd);
static void server_cmd_handler(std::filesystem::path model_path);
static void chat_cmd_handler(std::filesystem::path model_path);
static void llama_srv_cmd_handler(int argc, char * argv[]);

// global variable
static xoptions xoptions_;
std::filesystem::path home_path;
std::filesystem::path app_data_path;
common_params cparams_emb;

using namespace av_llm;

int main(int argc, char ** argv)
{
    {
#ifdef _WIN32
        home_path = std::getenv("USERPROFILE");
        if (home_path.empty())
            home_path = std::getenv("APPDATA"); // Fallback to APPDATA
#else
        home_path = std::getenv("HOME") ? std::filesystem::path(std::getenv("HOME")) : std::filesystem::path();
#endif
    }

    if (home_path.empty())
    {
        AVLLM_LOG_ERROR("%s: \n", "could not find the homepath");
        exit(1);
    }

    app_data_path = home_path / ".av_llm";
    AVLLM_LOG_DEBUG("home-path: %s -- app-path: %s \n", home_path.generic_string().c_str(), app_data_path.generic_string().c_str());

    pre_config_model_init();

    auto process_chat_or_serve = [](std::function<void(std::string)> chat_or_serve_func) -> int { // handle the syntax
        if (xoptions_.model_url_or_alias != "")
        {
            {
                // process if the .gguf file
                // AVLLM_LOG_DEBUG("%s:%d - %s \n", __func__, __LINE__, xoptions_.model_url_or_alias.c_str());
                std::filesystem::path model_filename = xoptions_.model_url_or_alias;
                if (model_filename.extension() == ".gguf")
                {
                    AVLLM_LOG_DEBUG("%s:%d - %s:%s \n", __func__, __LINE__, app_data_path.generic_string().c_str(),
                                    model_filename.c_str());

                    std::optional<std::filesystem::path> model_path = std::filesystem::is_regular_file(model_filename)
                        ? model_filename
                        : std::filesystem::is_regular_file(app_data_path / model_filename)
                        ? std::optional<std::filesystem::path>(app_data_path / model_filename)
                        : std::nullopt;

                    if (model_path.has_value())
                    {
                        AVLLM_LOG_DEBUG("%s:%d - %s \n", __func__, __LINE__, model_path.value().generic_string().c_str());
                        chat_or_serve_func(model_path.value().generic_string());
                    }
                    AVLLM_LOG_DEBUG("%s:%d \n", __func__, __LINE__);

                    return 0;
                }
            }

            {
                // handle precofig model
                if (std::string url = pre_config_model[xoptions_.model_url_or_alias]; url != "")
                {
                    AVLLM_LOG_DEBUG("%s:%d model url: %s\n", __func__, __LINE__, url.c_str());

                    std::filesystem::path model_filename = [&url]() -> std::filesystem::path {
                        auto last_slash = url.find_last_of("/");
                        if (last_slash != std::string::npos)
                            return url.substr(last_slash + 1);
                        return "";
                    }();

                    if (not std::filesystem::is_regular_file(app_data_path / model_filename))
                    { // do not have file in local, donwload it
                        AVLLM_LOG_DEBUG("%s:%d does NOT have model. So download it: %s from url: %s \n", __func__, __LINE__,
                                        model_filename.generic_string().c_str(), url.c_str());

                        downnload_file_and_write_to_file(url, model_filename);
                    }
                    else
                        AVLLM_LOG_DEBUG("%s:%d have model at:  %s \n", __func__, __LINE__, model_filename.generic_string().c_str());

                    chat_or_serve_func((app_data_path / model_filename).generic_string());

                    return 0;
                }
            }
        }
        else
        {
            chat_or_serve_func("");
        }

        return 0;
    };

    CLI::App app{ "av_llm - CLI program" };
    bool version_flag = false;
    app.add_flag("-v,--version", version_flag, "show version");

    // context
    app.add_option("--ctx", xoptions_.n_ctx, "Number of context")->default_val(std::to_string(xoptions_.n_ctx));
    app.add_option("--batch", xoptions_.n_batch, "Number of batch")->default_val(std::to_string(xoptions_.n_batch));

    //
    app.add_option("--ngl", xoptions_.ngl, "Number of context")->default_val(std::to_string(xoptions_.ngl));

    // sampling options
    app.add_option("--repeat-penalty", xoptions_.repeat_penalty, "Reapeat penanty")
        ->default_val(std::to_string(xoptions_.repeat_penalty));

    // ---- MODEL command ----
    auto model = app.add_subcommand("model", "Model operations");

    // model ls subcommand
    auto model_ls = model->add_subcommand("ls", "List all models");

    // model pull subcommand
    auto model_pull = model->add_subcommand("pull", "Pull a model (url: string)");
    model_pull->add_option("url-or-alias", xoptions_.model_url_or_alias, "Model URL")->required();

    // model del subcommand
    auto model_del = model->add_subcommand("del", "Delete a model (model: string)");
    model_del->add_option("url-or-alias", xoptions_.model_url_or_alias, "Model name")->required();

    // ---- CHAT command ----
    auto chat = app.add_subcommand("chat", "Start an interactive chat");
    chat->add_option("url-or-alias", xoptions_.model_url_or_alias, "Model path")->required();

    // ---- SERVE command ----
    auto serve = app.add_subcommand("serve", "Serve model");
    serve->add_option("-p,--port", xoptions_.port, "Serve port");
    serve->add_option("--emb-model", xoptions_.model_path_emb, "Embedding Model path");
    serve->add_option("url-or-alias", xoptions_.model_url_or_alias, "Model path");

    // -- llama comand ----
    auto llama = app.add_subcommand("llama", "LLAMA server command");
    llama->allow_extras();
    CLI11_PARSE(app, argc, argv);
    if (version_flag)
    {
        std::cout << "av_llm version 0.1.0" << std::endl;
        return 0;
    }

    // ---- MODEL logic ----
    if (*model)
    {
        if (*model_ls)
        {
            model_cmd_handler("ls");
        }
        else if (*model_pull)
        {
            model_cmd_handler("pull");
        }
        else if (*model_del)
        {
            model_cmd_handler("del");
        }
        else
        {
            std::cerr << "Specify a model subcommand: ls, pull <url>, or del <model>\n";
            model->help();
        }
        return 0;
    }

    // ---- CHAT logic ----
    if (*chat)
    {
        process_chat_or_serve(chat_cmd_handler);
        return 0;
    }
    // ---- SERVE logic ----
    if (*serve)
    {
        process_chat_or_serve(server_cmd_handler);
        return 0;
    }

    // ---- LLAMA logic ----
    if (*llama)
    {
        llama_server_main(argc - 1, &argv[1]);
        return 0;
    }

#if 0
    if (argc > 1)
    {
        process_chat_or_serve(argv[1], chat_cmd_handler);
    }
#endif

    std::cout << app.help() << std::endl;
}

static void model_cmd_handler(std::string sub_cmd)
{
    std::filesystem::create_directories(app_data_path);

    static auto get_file_name_from_url = [](const std::string url) -> std::string {
        auto last_slash = url.find_last_of('/');

        if (last_slash == std::string::npos)
            return url;

        return url.substr(last_slash + 1);
    };

    auto model_print_header = []() {
        std::cout << std::left << std::setw(70) << "|Model path" << std::setw(1) << '|' << "Size" << '\n';
        std::cout << std::string(78, '-') << '\n';
    };

    auto model_print = [](const std::filesystem::path & model_path, const std::uintmax_t & size) {
        std::cout << std::setw(70) << "|" + model_path.generic_string() << std::setw(1) << '|' << human_readable{ size } << "\n";
    };

    auto model_print_footer = []() { std::cout << std::string(77, '-') << std::endl; };

    auto model_pull = []() {
        AVLLM_LOG_DEBUG("%s: with argument: %s \n", "model_pull", xoptions_.model_url_or_alias.c_str());

        std::string url           = pre_config_model[xoptions_.model_url_or_alias] == "" ? xoptions_.model_url_or_alias
                                                                                         : pre_config_model[xoptions_.model_url_or_alias];
        std::string out_file_path = [](const std::string & url) -> std::string {
            auto last_slash = url.find_last_of('/');
            if (last_slash == std::string::npos)
                return url;
            return url.substr(last_slash + 1);
        }(url);
        downnload_file_and_write_to_file(url, out_file_path);
    };

    auto model_ls = [&model_print_header, &model_print, &model_print_footer]() {
        std::vector<std::filesystem::path> search_paths = {
            app_data_path // i.e. ~/.av_llm
        };

        model_print_header();

        for (const auto & search_path : search_paths)
        {
            if (std::filesystem::exists(search_path))
            {
                for (const auto & entry : std::filesystem::directory_iterator(search_path))
                {
                    if (entry.is_regular_file() && entry.path().extension() == ".gguf")
                    {
                        // Format the displayed path based on which directory it's from
                        model_print(entry.path(), entry.file_size());
                    }
                }
            }
        }

        model_print_footer();
    };

    auto model_del = []() { std::filesystem::remove(app_data_path / xoptions_.model_url_or_alias); };

    if (sub_cmd == "pull")
        model_pull();
    else if (sub_cmd == "del")
        model_del();
    else if (sub_cmd == "ls")
        model_ls();
};

static void chat_cmd_handler(std::filesystem::path model_path)
{
    bool silent = true;
    ggml_backend_load_all();

    // model initialized
    llama_model_ptr model = [&model_path]() -> llama_model_ptr {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = xoptions_.ngl;
        return llama_model_ptr(llama_model_load_from_file(model_path.generic_string().c_str(), model_params));
    }();
    if (!model)
    {
        AVLLM_LOG_ERROR("%s: error: unable to load model\n", __func__);
        return;
    }

    // context initialize
    llama_context_ptr ctx = [&model]() -> llama_context_ptr {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.no_perf              = false;
        ctx_params.n_ctx                = xoptions_.n_ctx;
        ctx_params.n_batch              = xoptions_.n_batch;
        return llama_context_ptr(llama_init_from_model(model.get(), ctx_params));
    }();
    if (!ctx)
    {
        AVLLM_LOG_ERROR("%s: error: failed to create the llama_context\n", __func__);
        return;
    }

    // initialize the sampler
    llama_sampler_ptr smpl = []() {
        auto sparams    = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        auto smpl       = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        return llama_sampler_ptr(smpl);
    }();
    if (!smpl)
    {
        AVLLM_LOG_ERROR("%s: error: could not create sampling\n", __func__);
        return;
    }
    if (!silent)
    {
        llama_sampler_print(smpl.get());
    }

    const char * chat_tmpl = llama_model_chat_template(model.get(), /* name */ nullptr);
    if (chat_tmpl == nullptr)
    {
        AVLLM_LOG_ERROR("%s: error: could no accept the template is null\n", __func__);
        return;
    }
    std::vector<llama_chat_message> chat_messages;
    std::vector<char> chat_message_output(llama_n_ctx(ctx.get()));
    int chat_message_start = 0;
    int chat_message_end   = 0;

    while (true)
    {

        std::string input_msg;
        { // get input string and apply template

            bool is_sys = chat_messages.size() == 0;
            std::cout << "\n" << (is_sys ? "system >" : "user   >");

            std::getline(std::cin, input_msg);
            if (input_msg.empty())
                break;

            chat_messages.push_back({ strdup(is_sys ? "system" : "user"), strdup(input_msg.c_str()) });

            int len = llama_chat_apply_template(chat_tmpl, chat_messages.data(), chat_messages.size(), true,
                                                chat_message_output.data(), chat_message_output.size());
            if (len > (int) chat_message_output.size())
            {
                chat_message_output.resize(len);
                len = llama_chat_apply_template(chat_tmpl, chat_messages.data(), chat_messages.size(), true,
                                                chat_message_output.data(), chat_message_output.size());
            }

            if (len < 0)
            {
                AVLLM_LOG_ERROR("%s: error: failed to apply chat template", __func__);
                return;
            }

            chat_message_end = len;
        }
        std::string prompt(chat_message_output.begin() + chat_message_start, chat_message_output.begin() + chat_message_end);

        chat_message_start = chat_message_end;

        llama_token new_token;
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        // const llama_vocab * vocab = llama_model_get_vocab(model);

        bool is_first       = llama_kv_self_used_cells(ctx.get()) == 0;
        int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);

        if (llama_tokenize(vocab, prompt.data(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0)
        {
            AVLLM_LOG_ERROR("%s: failed to tokenize the prompt \n", __func__);
            return;
        }

        if (!silent)
        { // print token
            std::cout << "\ntokens: \n";
            int cnt = 0;
            for (auto token : prompt_tokens)
            {
                char buf[120];
                int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
                if (n < 0)
                {
                    AVLLM_LOG_ERROR("%s: error: failed to tokenize \n", __func__);
                    return;
                }
                std::string s(buf, n);
                std::cout << s;
                cnt++;
            }
            std::cout << "end: " << cnt << "\n";
        }

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        while (true)
        {
            int n_ctx      = llama_n_ctx(ctx.get());
            int n_ctx_used = llama_kv_self_used_cells(ctx.get());

            if (n_ctx_used + batch.n_tokens > n_ctx)
            {
                AVLLM_LOG_WARN("%s: the context is exceeded. \n", __func__);
                return;
            }

            if (llama_decode(ctx.get(), batch))
            {
                AVLLM_LOG_ERROR("%s : failed to eval, return code %d\n", __func__, 1);
                return;
            }

            new_token = llama_sampler_sample(smpl.get(), ctx.get(), -1);
            if (llama_vocab_is_eog(vocab, new_token))
            {
                break;
            }

            char buf[100];
            int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                AVLLM_LOG_ERROR("%s, failed to convert a token \n", __func__);
                return;
            }

            std::string out(buf, n);
            std::cout << out;
            batch = llama_batch_get_one(&new_token, 1);
        }
    }

    if (false)
    {
        llama_perf_sampler_print(smpl.get());
        llama_perf_context_print(ctx.get());
    }

}; // end of chat handler

// Add this struct near the top, after includes and before usage
class chat_session_t : public http::base_data
{

public:
    chat_session_t(const chat_session_t &) = delete;
    chat_session_t(llama_context_ptr && _ctx) : http::base_data(), ctx(std::move(_ctx))
    {
        chat_message_start = 0;
        chat_message_end   = 0;
        chat_message_output.resize(llama_n_ctx(ctx.get()));
    }

    ~chat_session_t()
    {
        AVLLM_LOG_TRACE_SCOPE("~chat_session_t");
        if (ctx)
        {
            AVLLM_LOG_DEBUG("%s\n", "~chat_seesion_t is ended.");
        }
    }

public:
    llama_context_ptr ctx;
    std::vector<char> chat_message_output;
    int chat_message_start;
    int chat_message_end;

    const bool add_generation_prompt = true;
};

void server_cmd_handler(std::filesystem::path model_path)
{

    bool silent = false;
    llama_model_ptr model_general;
    common_chat_templates_ptr chat_templates;
    llama_sampler_ptr smpl_ptr;

    llama_model_ptr model_embedding;

    ggml_backend_load_all();

    if (xoptions_.model_url_or_alias != "")
    {
        llama_model_ptr model = [&model_path]() -> llama_model_ptr {
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers       = xoptions_.ngl;
            return llama_model_ptr(llama_model_load_from_file(model_path.generic_string().c_str(), model_params));
        }();
        if (!model)
        {
            AVLLM_LOG_ERROR("%s: error: unable to load model\n", __func__);
        }
        else
            model_general = std::move(model);

        // chat template
        chat_templates = [&]() {
            llama_model * model = model_general.get();
            auto chat_templates = common_chat_templates_init(model, "");
            try
            {
                common_chat_format_example(chat_templates.get(), false);
            } catch (const std::exception & e)
            {
                AVLLM_LOG_WARN("%s: Chat template parsing error: %s\n", __func__, e.what());
                AVLLM_LOG_WARN("%s: The chat template that comes with this model is not yet supported, falling back to chatml. "
                               "This may cause the "
                               "model to output suboptimal responses\n",
                               __func__);
                chat_templates = common_chat_templates_init(model, "chatml");
            }
            if (!silent)
            {
                AVLLM_LOG_INFO("%s: chat template: %s \n", __func__, common_chat_templates_source(chat_templates.get()));
                AVLLM_LOG_INFO("%s: chat example: %s \n", __func__,
                               common_chat_format_example(chat_templates.get(), false).c_str());
            }

            return chat_templates;
        }();

        // initialize the sampler
        // llama_sampler_ptr smpl;
        smpl_ptr = []() {
            auto sparams          = llama_sampler_chain_default_params();
            sparams.no_perf       = false;
            llama_sampler * smpl_ = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());
            return llama_sampler_ptr(smpl_);
        }();
        if (!smpl_ptr)
        {
            AVLLM_LOG_ERROR("%s: error: could not create sampling\n", __func__);
            return;
        }
        if (!silent)
        {
            llama_sampler_print(smpl_ptr.get());
        }
    }

    llama_context_params ctx_params = llama_context_default_params();

    auto me_llama_context_init = [&model_general, &ctx_params]() -> llama_context_ptr {
        llama_model * model = model_general.get();
        ctx_params.no_perf  = false;
        ctx_params.n_ctx    = xoptions_.n_ctx;
        ctx_params.n_batch  = xoptions_.n_batch;
        return llama_context_ptr(llama_init_from_model(model, ctx_params));
    };

    auto chat_session_get_n = [&](llama_sampler * smpl, chat_session_t & chat, std::vector<llama_token> & prompt_tokens,
                                  std::function<int(int, std::string)> func_) -> int {
        llama_token new_token;
        const llama_vocab * vocab = llama_model_get_vocab(model_general.get());

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        while (true)
        {
            int n_ctx      = llama_n_ctx(chat.ctx.get());
            int n_ctx_used = llama_kv_self_used_cells(chat.ctx.get());

            if (n_ctx_used + batch.n_tokens > n_ctx)
            {
                AVLLM_LOG_WARN("%s: the context is exceeded. \n", __func__);
                func_(-1, "");
                return -1;
            }

            if (llama_decode(chat.ctx.get(), batch))
            {
                AVLLM_LOG_ERROR("%s : failed to eval, return code %d\n", __func__, 1);
                func_(-1, "");
                return -1;
            }

            new_token = llama_sampler_sample(smpl, chat.ctx.get(), -1);
            if (llama_vocab_is_eog(vocab, new_token))
            {
                break;
            }

            char buf[100];
            int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                AVLLM_LOG_ERROR("%s, failed to convert a token \n", __func__);
                func_(-1, "");
                return -1;
            }

            std::string out(buf, n);
            // for (int i = 0; i < n; i++)
            //     printf("0x%04X, ", static_cast<unsigned char>(buf[i]));
            // printf("\n");

            if (func_(0, out) < 0)
            {
                AVLLM_LOG_WARN("%s, terminated by caller \n", __func__);
                break;
            }

            batch = llama_batch_get_one(&new_token, 1);
        }

        AVLLM_LOG_INFO("%s", "\n");
        return 0;
    };

    auto chat_session_get = [&](chat_session_t & chat, std::vector<llama_token> & prompt_tokens,
                                std::function<int(int, std::string)> func_) -> int {
        const llama_vocab * vocab = llama_model_get_vocab(model_general.get());

        return chat_session_get_n(smpl_ptr.get(), chat, prompt_tokens, func_);
    };

    auto chat_template_to_tokens =
        [&chat_templates, &model_general](chat_session_t & chat,
                                          const std::vector<llama_chat_message> & chat_messages) -> std::vector<llama_token> {
        llama_model * model       = model_general.get();
        const llama_vocab * vocab = llama_model_get_vocab(model);

        { // get input string and apply template

            auto chat_tmpl = common_chat_templates_source(chat_templates.get());

            int len = llama_chat_apply_template(chat_tmpl, chat_messages.data(), chat_messages.size(), true,
                                                chat.chat_message_output.data(), chat.chat_message_output.size());

            if (len > (int) chat.chat_message_output.size())
            {
                chat.chat_message_output.resize(len);
                len = llama_chat_apply_template(chat_tmpl, chat_messages.data(), chat_messages.size(), true,
                                                chat.chat_message_output.data(), chat.chat_message_output.size());
            }

            if (len < 0)
            {
                AVLLM_LOG_ERROR("%s: error: failed to apply chat template", __func__);
                return {};
            }

            chat.chat_message_end = len;
        }
        std::string prompt(chat.chat_message_output.begin() + chat.chat_message_start,
                           chat.chat_message_output.begin() + chat.chat_message_end);

        llama_token new_token;
        // const llama_vocab * vocab = llama_model_get_vocab(model);

        int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);

        if (llama_tokenize(vocab, prompt.data(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
        {
            AVLLM_LOG_ERROR("%s: failed to tokenize the prompt \n", __func__);
            return {};
        }
        return prompt_tokens;
    };

    // load embedding model if any
    if (xoptions_.model_path_emb != "")
    {
        std::vector<const char *> argvv = { "av_llm", "--pooling", "last" };

        if (common_params_parse(argvv.size(), (char **) argvv.data(), cparams_emb, LLAMA_EXAMPLE_EMBEDDING))
        {
            cparams_emb.embedding    = true;
            cparams_emb.n_batch      = (cparams_emb.n_batch < cparams_emb.n_ctx) ? cparams_emb.n_ctx : cparams_emb.n_batch;
            cparams_emb.n_ubatch     = cparams_emb.n_batch;
            cparams_emb.n_gpu_layers = xoptions_.ngl;
            // cparams_emb.
            llama_model_ptr model = []() {
                llama_model_params mparams = common_model_params_to_llama(cparams_emb);
                return llama_model_ptr(llama_load_model_from_file(xoptions_.model_path_emb.c_str(), mparams));
            }();
            GGML_ASSERT(model && "Can not initialize model");
            if (!model)
            {
                AVLLM_LOG_ERROR("%s: error: unable to load model\n", __func__);
            }
            else
                model_embedding = std::move(model);
        }
    }

    http::route route_;

    { // embedded web
        // legacy api
        struct handle_static_file
        {
            handle_static_file(std::string _file_path, std::string _content_type)
            {
                file_path    = _file_path;
                content_type = _content_type;
            }
            void operator()(std::shared_ptr<http::response> res)
            {
                AVLLM_LOG_TRACE_SCOPE("handle_static_file")
                if (not std::filesystem::exists(std::filesystem::path(file_path)))
                {
                    HTTP_SEND_RES_AND_RETURN(res, http::status_code::not_found, "File not found");
                }

                res->set_header("Content-Type", content_type);

                std::ifstream infile(file_path);
                std::stringstream str_stream;
                str_stream << infile.rdbuf();
                res->set_content(str_stream.str(), content_type);
                res->end();
            }

            std::string file_path;
            std::string content_type;
        };

        static auto preflight = [](std::shared_ptr<http::response> res) {
            res->set_header("Access-Control-Allow-Origin", res->reqwest().get_header("origin"));
            res->set_header("Access-Control-Allow-Credentials", "true");
            res->set_header("Access-Control-Allow-Methods", "POST");
            res->set_header("Access-Control-Allow-Headers", "*");
            res->set_content("", "text/html");
            res->end();
        };

        static auto web_handler = [&](std::shared_ptr<http::response> res) -> void {
            if (res->reqwest().get_header("accept-encoding").find("gzip") == std::string::npos)
            {
                res->set_content("Error: gzip is not supported by this browser", "text/plain");
            }
            else
            {
                res->set_header("Content-Encoding", "gzip");
                // COEP and COOP headers, required by pyodide (python interpreter)
                res->set_header("Cross-Origin-Embedder-Policy", "require-corp");
                res->set_header("Cross-Origin-Opener-Policy", "same-origin");
                res->set_content(reinterpret_cast<const char *>(av_llm::index_html_gz), av_llm::index_html_gz_len,
                                 "text/html; charset=utf-8");
                // res->set_content(reinterpret_cast<std::>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
            }
            res->end();
        };

        route_.set_option_handler(preflight);
        route_.get("/", web_handler);
        route_.get("/index.html", web_handler);
    }

    { // models
        static auto handle_models = [&](std::shared_ptr<http::response> res) {
            AVLLM_LOG_TRACE_SCOPE("handle_models")

            openai::ModelList models;
            models.add_model(openai::Model("model-id-stuff", "model", 1686935003, "stuff-01"));
            models.add_model(openai::Model("model-id-stuff", "model", 1686935004, "stuff-02"));

            res->set_content(models.to_json().dump(4), MIMETYPE_JSON);
            res->end();
        };

        static auto handle_model_detail = [&](std::shared_ptr<http::response> res) -> void {
            AVLLM_LOG_TRACE_SCOPE("handle_model");
            std::string model_name = res->reqwest().get_param("model");
            if (model_name.empty())
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "Model is required");

            openai::Model model_info;
            model_info.object   = "model";
            model_info.id       = model_name;
            model_info.created  = std::time(0);
            model_info.owned_by = "avbl llm";

            // nlohmman json dump with pretty
            res->set_content(model_info.to_json().dump(4), MIMETYPE_JSON);
            res->end();
        };

        route_.get("/models", handle_models);
        route_.get("/v1/models", handle_models);
        route_.get("/models/{model}", handle_model_detail);
        route_.get("/v1/models/{model}", handle_model_detail);
    }

    // OpenAI API (chat, embedding)
    {

        static auto completions_chat_handler = [&](std::shared_ptr<http::response> res) -> void {
            AVLLM_LOG_TRACE_SCOPE("completions_chat_handler")
            AVLLM_LOG_DEBUG("%s: session-id: %" PRIu64 "\n", "completions_chat_handler", res->session_id());

            enum CHAT_TYPE
            {
                CHAT_TYPE_DEFAULT = 0,
                CHAT_TYPE_STREAM  = 1
            };

            CHAT_TYPE chat_type = CHAT_TYPE_DEFAULT;

            json body_ = json_parse(res->reqwest().body());
            if (body_.empty())
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "Empty request body");

            json messages_js = json_value(body_, "messages", json());
            if (messages_js.empty())
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "Missing or empty messages");

            if (body_ == json() or messages_js == json())
            {
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Invalid JSON format");
            }

            chat_type = json_value(body_, "stream", bool(false)) == true ? CHAT_TYPE_STREAM : CHAT_TYPE_DEFAULT;

            if (chat_type == CHAT_TYPE_DEFAULT or chat_type == CHAT_TYPE_STREAM)
            {
                // printf("[DEBUG] %s:%d \n", __func__, __LINE__);
                try
                {
                    if (!res->get_session_data())
                    {
                        AVLLM_LOG_INFO("%s: initialize context for session id: %" PRIu64 " \n", "completions_chat_handler",
                                       res->session_id());
                        llama_context_ptr p = me_llama_context_init();
                        if (p)
                        {
                            res->get_session_data() = std::make_unique<chat_session_t>(std::move(p));
                        }
                        else
                            throw std::runtime_error("can not initalize context");
                    }
                } catch (const std::exception & e)
                {
                    AVLLM_LOG_WARN("%s: can not initialize the context \n", __func__);
                    HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Failed to initialize context");
                }

                chat_session_t * chat_session = static_cast<chat_session_t *>(res->get_session_data().get());
                GGML_ASSERT(nullptr != chat_session);

                std::vector<llama_chat_message> chat_messages;
                std::string prompt;

                // extract promt
                for (const auto & msg : messages_js)
                {
                    std::string role = (msg.contains("role") and msg.at("role").is_string()) ? msg.at("role") : "";
                    if (role == "user" or role == "system" or role == "assistant")
                    {
                        std::string content = msg.at("content").get<std::string>();
                        chat_messages.push_back({ strdup(role.c_str()), strdup(content.c_str()) });
                        std::cout << role << ":" << content << std::endl;
                    }
                }

                if (chat_type == CHAT_TYPE_DEFAULT)
                { // default
                    std::string res_body;
                    auto get_text_hdl = [&](int rc, const std::string & text) -> int {
                        if (rc == 0)
                            res_body = res_body + text;

                        return 0;
                    };

                    auto tokens = chat_template_to_tokens(*chat_session, chat_messages);
                    if (tokens.size() == 0)
                    {
                        HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error,
                                                 "Tokenization failed - no tokens generated");
                    }

                    chat_session_get(*chat_session, tokens, get_text_hdl);
                    res->set_content(res_body);
                    res->end();
                }
                else
                { // stream
                    res->event_source_start();
                    auto get_text_hdl = [&](int rc, const std::string & text) -> int {
                        auto is_all_printable = [](const std::string & s) -> bool {
                            return !std::any_of(s.begin(), s.end(), [](unsigned char c) { return !std::isprint(c); });
                        };

                        // TODO: why only pritable character
                        if (rc == 0 and is_all_printable(text))
                        {
                            res->chunk_write_async("data: " + oai_make_stream(text));
                        }

                        return 0;
                    };

                    auto tokens = chat_template_to_tokens(*chat_session, chat_messages);
                    if (tokens.size() == 0)
                    {
                        HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error,
                                                 "Tokenization failed - no tokens generated");
                    }

                    chat_session_get(*chat_session, tokens, get_text_hdl);

                    res->chunk_end_async();
                }
            }
            else
            {
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Unsupported chat type");
            }
        };

        static auto embedding_handler = [&](std::shared_ptr<http::response> res) -> void {
            // sanity check
            if (!model_embedding)
            {
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Embedding model not available");
            }
            llama_model * model = model_embedding.get();

            json body_js = json_parse(res->reqwest().body());

            if (body_js.empty())
            {
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Invalid request content");
            }

            // fix: the sentence-level embedding
            // context
            llama_context_ptr ctx = [&]() {
                llama_context_params cparams = common_context_params_to_llama(cparams_emb);
                return llama_context_ptr(llama_init_from_model(model, cparams));
            }();
            GGML_ASSERT(ctx && "Can not initilize the context");
            if (!ctx)
            {
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Failed to initialize embedding context");
            }

            std::string input = json_value(body_js, "input", std::string());
            if (input == "")
            {
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Empty input text");
            }
            // tokenize the prompt
            auto tokens = [&model, &data = input]() -> std::vector<llama_token> {
                const llama_vocab * const vocab = llama_model_get_vocab(model);
                GGML_ASSERT(vocab != nullptr);
                int n_token = -llama_tokenize(vocab, data.data(), data.size(), NULL, 0, true, true);
                std::vector<llama_token> tokens(n_token);
                if (llama_tokenize(vocab, data.data(), data.size(), tokens.data(), tokens.size(), true, true) < 0)
                    return {};
                return tokens;
            }();
            if (tokens.size() == 0)
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Failed to tokenize input");

            if (!silent)
            {
                const llama_vocab * const vocab = llama_model_get_vocab(model);
                llama_token_print(vocab, tokens);
            }

            const uint64_t n_batch = cparams_emb.n_batch;
            llama_batch batch      = llama_batch_init(n_batch, 0, 1);

            for (int pos = 0; pos < tokens.size(); pos++)
                common_batch_add(batch, tokens[pos], pos, { 0 }, true);

            if (!silent)
                llama_batch_print(&batch);

            if (llama_decode(ctx.get(), batch) < 0)
            {
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Failed to decode embedding batch");
            }

            int n_embd = llama_model_n_embd(model);
            std::vector<float> embeddings(n_embd);

            float * embd                         = llama_get_embeddings_seq(ctx.get(), 0);
            float * const out                    = embeddings.data();
            enum llama_pooling_type pooling_type = llama_pooling_type(ctx.get());
            common_embd_normalize(embd, out, n_embd, cparams_emb.embd_normalize);
            if (!silent)
            {
                for (int i = 0; i < std::min(3, n_embd); i++)
                    printf("%.6f ", *(embeddings.data() + i));
                printf("...");

                for (int i = 0; i < n_embd && i < 2; i++)
                    printf("%.6f ", *(embeddings.data() + n_embd - 1 - i));
            }
            json j = embeddings;
            res->set_content(j.dump(4));
            res->end();
        };

        // completion (it is only support stream, and non-stream)
        route_.post("/completions",
                    [](std::shared_ptr<http::response> res) { std::thread{ completions_chat_handler, res }.detach(); });
        route_.post("/v1/completions",
                    [](std::shared_ptr<http::response> res) { std::thread{ completions_chat_handler, res }.detach(); });
        route_.post("/chat/completions",
                    [](std::shared_ptr<http::response> res) { std::thread{ completions_chat_handler, res }.detach(); });
        route_.post("/v1/chat/completions",
                    [](std::shared_ptr<http::response> res) { std::thread{ completions_chat_handler, res }.detach(); });

        route_.post("/embeddings", [](std::shared_ptr<http::response> res) { embedding_handler(res); });
        route_.post("/v1/embeddings", [](std::shared_ptr<http::response> res) { embedding_handler(res); });
    }

    // infill, fim
    {
        static auto fim_handler = [&me_llama_context_init, &chat_session_get_n, &model_general,
                                   &ctx_params](std::shared_ptr<http::response> res) -> void {
            AVLLM_LOG_TRACE_SCOPE("fim handler");
            bool silent               = true;
            const llama_vocab * vocab = llama_model_get_vocab(model_general.get());
            if (!silent)
                AVLLM_LOG_INFO("%s \n", res->reqwest().body().c_str());

            // check model compatibility
            std::string is_err = [&vocab]() -> std::string {
                if (llama_vocab_fim_pre(vocab) == LLAMA_TOKEN_NULL)
                {
                    return "prefix token is missing. ";
                }
                if (llama_vocab_fim_suf(vocab) == LLAMA_TOKEN_NULL)
                {
                    return "suffix token is missing. ";
                }
                if (llama_vocab_fim_mid(vocab) == LLAMA_TOKEN_NULL)
                {
                    return "middle token is missing. ";
                }
                return "";
            }();
            if (not is_err.empty())
            {
                AVLLM_LOG_WARN("%s: %s \n", __func__, is_err.c_str());
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, is_err);
            }

            json body_js;
            [&body_js](const std::string & body) mutable {
                try
                {
                    body_js = json::parse(body);
                } catch (const std::exception & ex)
                {
                    body_js = {};
                }
            }(res->reqwest().body());

            if (body_js == json())
            { // not valid body_js
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Invalid JSON body");
            }

            is_err = [&body_js]() -> std::string {
                if (body_js.contains("prompt") && !body_js.at("prompt").is_string())
                    return "Do not accept the prompt (not string)";

                if (!body_js.contains("input_prefix"))
                    return R"("input_prefix" is required)";

                if (!body_js.contains("input_suffix"))
                    return R"("input_suffix" is required)";

                // if (body_js.contains("input_extra") && !body_js.at("input_extra").is_array())
                //    return "\"input_extra\" must be an array of {\"filename\": string, \"text\": string}";

                return "";
            }();

            if (!is_err.empty())
            {
                AVLLM_LOG_WARN("%s: %s \n", __func__, is_err.c_str());
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, is_err);
            }

            json input_extra = json_value(body_js, "input_extra", json::array());

            [&input_extra]() -> std::string {
                for (const auto & chunk : input_extra)
                {
                    // { "text": string, "filename": string }
                    if (!chunk.contains("text") || !chunk.at("text").is_string())
                        return "extra_context chunk must contain a \"text\" field with a string value";

                    // filename is optional
                    if (chunk.contains("filename") && !chunk.at("filename").is_string())
                        return "extra_context chunk's \"filename\" field must be a string";
                }
                return "";
            }();

            body_js["input_extra"] = input_extra; // default to empty array if it's not exist

            std::string prompt                          = json_value(body_js, "prompt", std::string());
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(vocab, prompt, false, true);

            std::string input_suffix = body_js.at("input_suffix").get<std::string>();
            std::string input_prefix = body_js.at("input_prefix").get<std::string>();

            if (input_prefix.empty())
            {
                res->end();
                AVLLM_LOG_WARN("%s \n", "input_prefix is empty");
                return;
            }

            if (!silent)
                AVLLM_LOG_INFO("prefix:\n%s\nsuffix:\n%s\n", input_prefix.c_str(), input_suffix.c_str());

            int n_predict   = json_value(body_js, "n_predict", 128);
            int top_k       = json_value(body_js, "top_k", 40);
            int top_p       = json_value(body_js, "top_p", 0.89);
            int temperature = json_value(body_js, "temperature", 1.0f);
            int32_t seed    = json_value(body_js, "seed", 4294967295);
            json samplers   = json_value(body_js, "samplers", json::array());

            auto tokens = format_infill(vocab, input_prefix, input_suffix, body_js.at("input_extra"), ctx_params.n_batch, n_predict,
                                        ctx_params.n_ctx, false, tokenized_prompts[0]);
            if (!silent)
            {
                llama_token_print(vocab, tokens);
            }

            try
            {
                if (!res->get_session_data())
                {
                    AVLLM_LOG_INFO("%s: initialize context for session id: %" PRIu64 " \n", "fim_handler", res->session_id());
                    llama_context_ptr p = me_llama_context_init();
                    if (p)
                        res->get_session_data() = std::make_unique<chat_session_t>(std::move(p));
                    else
                        throw std::runtime_error("can not initalize context");
                }
            } catch (const std::exception & e)
            {
                AVLLM_LOG_WARN("%s: can not initialize the context \n", __func__);
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Failed to initialize context");
            }

            chat_session_t * chat_session = static_cast<chat_session_t *>(res->get_session_data().get());

            if (tokens.size() == 0)
            {
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "No tokens generated");
            }

            std::string res_body;
            int decoded       = 0;
            auto get_text_hdl = [&](int rc, const std::string & text) -> int {
                if (decoded++ > n_predict)
                    return -1;
                if (rc == 0)
                    res_body = res_body + text;

                return 0;
            };

            // sampler
            // initialize the sampler
            llama_sampler_ptr smpl_ = [&vocab, &top_k, &top_p, &seed]() {
                auto sparams    = llama_sampler_chain_default_params();
                sparams.no_perf = false;
                auto smpl       = llama_sampler_chain_init(sparams);
                llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
                llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 20));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
                return llama_sampler_ptr(smpl);
            }();
            if (!smpl_)
            {
                AVLLM_LOG_ERROR("%s: error: could not create sampling\n", __func__);
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Could not create sampler");
            }
            if (!silent)
            {
                llama_sampler_print(smpl_.get());
            }

            {
                chat_session_get_n(smpl_.get(), *chat_session, tokens, get_text_hdl);
                json body_js;
                body_js["content"] = res_body;
                res->set_content(body_js.dump());
                res->end();
                return;
            }
        };

        route_.post("/fim", [](std::shared_ptr<http::response> res) { std::thread{ fim_handler, res }.detach(); });
        route_.post("/infill", [](std::shared_ptr<http::response> res) { std::thread{ fim_handler, res }.detach(); });
    }

    // Example of how to use the new async chunking API for streaming responses
    {
        static auto streaming_handler = [&me_llama_context_init, &chat_session_get_n, &model_general,
                                         &ctx_params](std::shared_ptr<http::response> res) -> void {
            AVLLM_LOG_TRACE_SCOPE("streaming handler");

            // Set up streaming response headers
            res->set_header("Content-Type", "text/event-stream");
            res->set_header("Cache-Control", "no-cache");
            res->set_header("Connection", "keep-alive");
            res->set_header("Access-Control-Allow-Origin", "*");

            // Start chunked response
            res->chunk_start_async([res](bool success) {
                if (!success)
                {
                    AVLLM_LOG_ERROR("Failed to start chunked response");
                    return;
                }

                // Example of streaming data
                std::string chunk_data = "data: {\"content\": \"Hello\"}\n\n";
                res->chunk_write_async(chunk_data, [res](bool success) {
                    if (!success)
                    {
                        AVLLM_LOG_ERROR("Failed to write chunk");
                        return;
                    }

                    // Write another chunk
                    std::string chunk_data2 = "data: {\"content\": \" World\"}\n\n";
                    res->chunk_write_async(chunk_data2, [res](bool success) {
                        if (!success)
                        {
                            AVLLM_LOG_ERROR("Failed to write second chunk");
                            return;
                        }

                        // End the streaming response
                        res->chunk_end_async([](bool success) {
                            if (!success)
                            {
                                AVLLM_LOG_ERROR("Failed to end chunked response");
                            }
                        });
                    });
                });
            });
        };

        route_.post("/stream", [](std::shared_ptr<http::response> res) { std::thread{ streaming_handler, res }.detach(); });
    }

    {
        // health handler
        static auto health_handler = [](std::shared_ptr<http::response> res) {
            res->set_header("Content-Type", "application/json");
            json body;
            body["status"]  = "OK";
            body["name"]    = "av_llm";
            body["version"] = "0.0.1-Preview";
            body["uptime"]  = 0;
            res->set_content(body.dump());
            res->end();
        };

        //
        route_.get("health", health_handler);
    }

    // registered route by post, get, ..
    // get        /
    // get        /index.html
    //
    // get				/models
    // get 				/v1/models
    //
    // post 			/completions
    // post 			/v1/completions
    // post				/v1/chat/completions
    //
    // post 			/infill
    // post 			/fim
    //
    // get 				/health

    AVLLM_LOG_INFO("Server is started at http://127.0.0.1:%d\n", xoptions_.port);

    http::start_server(xoptions_.port, route_);

    // llama_model_free(model);
};
