/*
 * Copyright (c) 2024-2024 Harry Le (avble.harry at gmail dot com)

It can be used, modified.
*/

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama-cpp.h"
#include "llama.h"
#include "log.hpp"

#include "av_connect.hpp"
#include "model.hpp"
#include "openai.hpp"
namespace av_llm {
#include "index.html.gz.hpp"
}
#include "utils.hpp"

#include <CLI/CLI.hpp>
#include <inttypes.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
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

// console color
// Text color
#define CONSOLE_COLOR_BLACK "\033[30m"
#define CONSOLE_COLOR_RED "\033[31m"
#define CONSOLE_COLOR_GREEN "\033[32m"
#define CONSOLE_COLOR_YELLOW "\033[33m"
#define CONSOLE_COLOR_BLUE "\033[34m"
#define CONSOLE_COLOR_MAGENTA "\033[35m"
#define CONSOLE_COLOR_CYAN "\033[36m"
#define CONSOLE_COLOR_WHITE "\033[37m"

// Bold text
#define CONSOLE_BOLD "\033[1m"

// Reset all attributes
#define CONSOLE_RESET "\033[0m"

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

#ifdef NDEBUG
    llama_log_set(
        [](enum ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_DEBUG)
            {
                fprintf(stderr, "%s", text);
            }
        },
        nullptr);
#endif
    if (home_path.empty())
    {
        AVLLM_LOG_ERROR("%s: \n", "could not find the homepath");
        exit(1);
    }

    app_data_path = home_path / ".av_llm";
    AVLLM_LOG_DEBUG("home-path: %s -- app-path: %s \n", home_path.generic_string().c_str(), app_data_path.generic_string().c_str());

    pre_config_model_init();

    auto execute_char_or_serve = [](std::function<void(std::string)> chat_or_serve_func) -> int { // handle the syntax
        if (xoptions_.model_url_or_alias != "")
        {
            {
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
    app.add_option("--npredict", xoptions_.n_predict, "Number of predict")->default_val(std::to_string(xoptions_.n_predict));
    app.add_flag("--jinja", xoptions_.jinja, "Use jinja template for chat format");

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
    serve->add_option("--np", xoptions_.n_parallel, "Number of parallel requests");
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
            auto app2 = model->help();
        }
        return 0;
    }

    // ---- CHAT logic ----
    if (*chat)
    {
        execute_char_or_serve(chat_cmd_handler);
        return 0;
    }
    // ---- SERVE logic ----
    if (*serve)
    {
        execute_char_or_serve(server_cmd_handler);
        return 0;
    }

    // ---- LLAMA logic ----
    if (*llama)
    {
        llama_server_main(argc - 1, &argv[1]);
        return 0;
    }

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
        AVLLM_LOG_DEBUG("%s", "sampler:\n");
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

class session_chat_message : public http::base_data
{

    std::vector<char> message;
    int start;
    int end;
};

// helper
std::string model_oaicompact_to_text(const llama_model * model, const nlohmann::ordered_json & oai_js)
{
    std::string result;

    const std::string str_messages = oai_js.at("messages").dump();

    std::string bos_token = "";
    std::string eos_token = "";

    std::vector<common_chat_msg> messages = common_chat_msgs_parse_oaicompat(str_messages);

    const std::string str_tools = json_value(oai_js, "tools", std::string()); // oai_js.at("tools").dump();
    std::vector<common_chat_tool> tools =
        str_tools.empty() ? std::vector<common_chat_tool>() : common_chat_tools_parse_oaicompat(str_tools);

    const auto add_generation_prompt = true;

    common_chat_templates_inputs inputs;
    inputs.use_jinja             = xoptions_.jinja;
    inputs.messages              = messages;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.tools                 = tools;

    std::string template_jinja;
    auto tmpls = common_chat_templates_init(model, template_jinja.c_str(), bos_token, eos_token);
    try
    {
        result = common_chat_templates_apply(tmpls.get(), inputs).prompt;
    } catch (const std::exception & e)
    {
        AVLLM_LOG_WARN("%s: Chat template parsing error: %s\n", __func__, e.what());
    }
    return result;
};

int context_gen_text_until_eog(llama_context * ctx, std::vector<llama_token> & prompt_tokens,
                               std::function<int(int, const std::string &)> func_, llama_sampler * smpl)
{
    llama_token new_token;
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    while (true)
    {
        int n_ctx      = llama_n_ctx(ctx);
        int n_ctx_used = llama_kv_self_used_cells(ctx);

        if (n_ctx_used + batch.n_tokens > n_ctx)
        {
            AVLLM_LOG_WARN("%s: the context is exceeded. \n", __func__);
            func_(-1, "");
            return -1;
        }

        if (llama_decode(ctx, batch))
        {
            AVLLM_LOG_ERROR("%s : failed to eval, return code %d\n", __func__, 1);
            func_(-1, "");
            return -1;
        }

        new_token = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token))
        {
            func_(-1, "");
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
        // std::cout << out;

        if (func_(0, out) < 0)
        {
            AVLLM_LOG_WARN("%s, terminated by caller \n", __func__);
            return 0;
        }

        batch = llama_batch_get_one(&new_token, 1);
    }

    return 0; // return -1 means end of the ctx_session_ session
};

void server_cmd_handler(std::filesystem::path model_path)
{

#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    struct model_general_t
    {

        model_general_t(std::string _model_path) : model_path(_model_path) {}

        void init()
        {

            {
                llama_model_params model_params = llama_model_default_params();
                model_params.n_gpu_layers       = xoptions_.ngl;
                model_ptr                       = llama_model_ptr(llama_model_load_from_file(model_path.c_str(), model_params));
            }
            if (!model_ptr)
                return;

            {
                llama_model * model = model_ptr.get();
                chat_templates_ptr  = common_chat_templates_init(model, "");
                try
                {
                    common_chat_format_example(chat_templates_ptr.get(), false);
                } catch (const std::exception & e)
                {
                    AVLLM_LOG_WARN("%s: Chat template parsing error: %s\n", __func__, e.what());
                    AVLLM_LOG_WARN("%s: The chat template that comes with this model is not yet supported, falling back to chatml. "
                                   "This may cause the "
                                   "model to output suboptimal responses\n",
                                   __func__);
                    chat_templates_ptr = common_chat_templates_init(model, "chatml");
                }
                {
                    AVLLM_LOG_DEBUG("%s: chat template: %s \n", __func__, common_chat_templates_source(chat_templates_ptr.get()));
                    AVLLM_LOG_DEBUG("%s: chat example: %s \n", __func__,
                                    common_chat_format_example(chat_templates_ptr.get(), false).c_str());
                }
            }

            // initialize the sampler
            {
                auto sparams          = llama_sampler_chain_default_params();
                sparams.no_perf       = false;
                llama_sampler * smpl_ = llama_sampler_chain_init(sparams);
                llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());
                sampler_default_ptr = llama_sampler_ptr(smpl_);
            }

            if (!sampler_default_ptr)
            {
                AVLLM_LOG_ERROR("%s: error: could not create sampling\n", __func__);
                return;
            }

            {
                int num_context = std::max(1, xoptions_.n_parallel);
                num_context     = std::min(num_context, 16);
                for (int i = 0; i < num_context; i++)
                {
                    llama_context_params ctx_params = llama_context_default_params();
                    llama_model * model             = model_ptr.get();
                    ctx_params.no_perf              = false;
                    ctx_params.n_ctx                = xoptions_.n_ctx;
                    ctx_params.n_batch              = xoptions_.n_batch;
                    // ctx_params.flash_attn           = true;
                    contexts.emplace_back(llama_init_from_model(model_ptr.get(), ctx_params));
                }
            }

            initialized = true;
        }

        const llama_model * get_model() const
        {
            if (!model_ptr)
            {
                AVLLM_LOG_ERROR("%s: error: model is not initialized\n", __func__);
                return nullptr;
            }
            return model_ptr.get();
        }

        llama_sampler * get_sampler()
        {
            if (!sampler_default_ptr)
            {
                AVLLM_LOG_ERROR("%s: error: sampler is not initialized\n", __func__);
                return nullptr;
            }
            return sampler_default_ptr.get();
        }

        llama_context * get_context(int idx)
        {
            if (idx < 0 || idx >= (int) contexts.size())
            {
                AVLLM_LOG_ERROR("%s: error: invalid context index %d\n", __func__, idx);
                return nullptr;
            }
            return contexts[idx].get();
        }

        int get_n_ctx() const { return contexts.size(); }

        std::vector<llama_token> model_string_to_tokens(const std::string & str)
        {
            llama_model * model = model_ptr.get();
            auto tokens         = [&model, &str]() -> std::vector<llama_token> {
                const llama_vocab * vocab = llama_model_get_vocab(model);
                int n_token               = -llama_tokenize(vocab, str.data(), str.size(), NULL, 0, true, true);
                std::vector<llama_token> tokens(n_token);
                if (llama_tokenize(vocab, str.data(), str.size(), tokens.data(), tokens.size(), true, true) < 0)
                    return {};
                return tokens;
            }();
            return tokens;
        };

        bool is_initialized() const { return initialized; }

        llama_model_ptr model_ptr                    = nullptr;
        common_chat_templates_ptr chat_templates_ptr = nullptr;
        llama_sampler_ptr sampler_default_ptr        = nullptr;
        std::string model_path;
        bool initialized = false;
        // llama_context_ptr ctx_ptr;

        std::vector<llama_context_ptr> contexts;

    } model_general(model_path.generic_string());

    llama_model_ptr model_embedding;

    ggml_backend_load_all();

    if (xoptions_.model_url_or_alias != "")
    {
        model_general.init();
        if (!model_general.is_initialized())
            AVLLM_LOG_WARN("%s: error: unable to load model\n", __func__);
    }

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

    // embedded web
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

    // oai - models
    static auto handle_models = [](std::shared_ptr<http::response> res) {
        AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(),
                                                    res->reqwest().request_id(), "handle_models")
                                  .c_str())

        openai::ModelList models;
        models.add_model(openai::Model("model-id-stuff", "model", static_cast<int64_t>(time(0)), "stuff-01"));

        res->set_content(models.to_json().dump(4), MIMETYPE_JSON);
        res->end();
    };

    static auto handle_model_detail = [](std::shared_ptr<http::response> res) -> void {
        AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(),
                                                    res->reqwest().request_id(), "handle_model")
                                  .c_str())
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

    static auto api_tags_handler = [&model_general](std::shared_ptr<http::response> res) {
        const llama_model * model = model_general.get_model();
        const llama_context * ctx = model_general.get_context(0);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        std::string general_name;
        general_name.reserve(64);
        std::string base_name;
        base_name.reserve(64);

        llama_model_meta_val_str(model, "general.name", general_name.data(), general_name.capacity());
        llama_model_meta_val_str(model, "base.name", base_name.data(), base_name.capacity());
        int n_ctx_train = llama_model_n_ctx_train(model);

        json resp = { { "models",
                        { { { "name", general_name },
                            { "model", base_name },
                            { "modified_at", "" },
                            { "size", "" },
                            { "digest", "" },
                            { "type", "model" },
                            { "description", "" },
                            { "tags", { "" } },
                            { "capabilities", { "completion" } },
                            { "parameters", "" },
                            { "details",
                              { { "parent_model", "" },
                                { "format", "gguf" },
                                { "family", "" },
                                { "families", { "" } },
                                { "parameter_size", "" },
                                { "quantization_level", "" } } } } } },
                      { "object", "list" },
                      { "data",
                        { { { "id", general_name },
                            { "object", "model" },
                            { "created", 1752894428 },
                            { "owned_by", "llamacpp" },
                            { "meta",
                              { { "vocab_type", static_cast<int>(llama_vocab_type(vocab)) },
                                { "n_vocab", llama_vocab_n_tokens(vocab) },
                                { "n_ctx_train", n_ctx_train },
                                { "n_embd", 2048 },
                                { "n_params", 3085938688 },
                                { "size", 3279519744 } } } } } } };
        res->set_header("Content-Type", "application/json");
        res->set_content(resp.dump(4));
        res->end();
    };

    auto api_show = [&model_general](std::shared_ptr<http::response> res) {
        AVLLM_LOG_TRACE_SCOPE(
            av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(), res->reqwest().request_id(), "api_show")
                .c_str())

        const llama_model * model = model_general.get_model();
        int n_ctx_train           = llama_model_n_ctx_train(model);

        json resp = { { "template", "" },
                      { "model_info", { { "llama.context_length", n_ctx_train } } },
                      { "modelfile", "" },
                      { "parameters", "" },
                      { "details",
                        { { "parent_model", "" },
                          { "format", "gguf" },
                          { "family", "" },
                          { "families", { "" } },
                          { "parameter_size", "" },
                          { "quantization_level", "" } } },
                      { "capabilities", { "completion" } } };
        res->set_header("Content-Type", "application/json");
        res->set_content(resp.dump(4));
        res->end();
    };

    static auto responses_handler = [&model_general](std::shared_ptr<http::response> res, int ctx_idx) -> void {
        AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(),
                                                    res->reqwest().request_id(), "responses_handler")
                                  .c_str())

        json body_ = json_parse(res->reqwest().body());

        if (body_.empty())
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "invalid json");

        // print the body for debugging
        AVLLM_LOG_DEBUG("[%05" PRIu64 "] [%05" PRIu64 "] body: %s", res->session_id(), res->reqwest().request_id(),
                        body_.dump(4).c_str());

        std::string model_name  = json_value(body_, "model", std::string("model"));
        std::string input       = json_value(body_, "input", std::string(""));
        std::string instruction = json_value(body_, "instructions", std::string("continue"));

        if (input.empty())
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "Input is required");

        llama_context * ctx       = model_general.get_context(ctx_idx);
        const llama_model * model = llama_get_model(ctx);
        llama_sampler * smpl      = model_general.get_sampler();
        std::string id            = string_generate_random(64);
        int time                  = std::time(0);

        if (instruction == "restart")
            llama_memory_clear(llama_get_memory(ctx), true);

        // tokenize the prompt
        auto prompt_tokens = model_general.model_string_to_tokens(input);

        if (prompt_tokens.size() == 0)
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "Tokenization failed - no tokens generated");

        // write above struct in lambda function
        std::string gen_text;
        uint32_t completion_tokens  = 0;
        uint32_t prompt_tokens_size = static_cast<uint32_t>(prompt_tokens.size());
        auto gen_text_hdl           = [prompt_tokens_size, &gen_text, &completion_tokens](int rc, const std::string & text) -> int {
            if (completion_tokens >= xoptions_.n_predict)
                return -1; // end of generation
            if (rc == 0)
                gen_text += text;

            completion_tokens++;
            return 0; // continue generation
        };

        context_gen_text_until_eog(ctx, prompt_tokens, std::ref(gen_text_hdl), smpl);
        json res_body = {
            { "id", "resp_" + id },
            { "object", "response" },
            { "created_at", time },
            { "status", "completed" },
            { "error", nullptr },
            { "incomplete_details", nullptr },
            { "instructions", nullptr },
            { "max_output_tokens", nullptr },
            { "model", model_name },
            { "output",
              { { { "type", "message" },
                  { "id", "msg_" + id },
                  { "status", "completed" },
                  { "role", "assistant" },
                  { "content",
                    { { { "type", "output_text" }, { "text", gen_text }, { "annotations", nlohmann::json::array() } } } } } } },
            { "parallel_tool_calls", true },
            { "previous_response_id", nullptr },
            { "reasoning", { { "effort", nullptr }, { "summary", nullptr } } },
            { "store", true },
            { "temperature", 1.0 },
            { "text", { { "format", { { "type", "text" } } } } },
            { "tool_choice", "auto" },
            { "tools", nlohmann::json::array() },
            { "top_p", 1.0 },
            { "truncation", "disabled" },
            { "usage",
              { { "input_tokens", 36 },
                { "input_tokens_details", { { "cached_tokens", 0 } } },
                { "output_tokens", 87 },
                { "output_tokens_details", { { "reasoning_tokens", 0 } } },
                { "total_tokens", 123 } } },
            { "user", nullptr },
            { "metadata", nlohmann::json::object() }
        };

        res->set_content(res_body.dump(4));
        res->endend();
    };

    // oai (completions, chat completions, embedding)
    static auto completions_handler = [&model_general](std::shared_ptr<http::response> res, int ctx_idx) -> void {
        AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(),
                                                    res->reqwest().request_id(), "completions_handler")
                                  .c_str())

        llama_context * ctx       = model_general.get_context(ctx_idx);
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_sampler * smpl      = model_general.get_sampler();

        json body_ = json_parse(res->reqwest().body());
        if (body_.empty())
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "invalid json");

        std::string model_name = json_value(body_, "model", std::string("model"));
        int max_tokens         = json_value(body_, "max_tokens", int(1024));
        std::string prompt     = json_value(body_, "prompt", std::string());
        bool is_stream         = json_value(body_, "stream", bool(false));
        int64_t temperature    = json_value(body_, "temperature", int64_t(0));

        AVLLM_LOG_DEBUG("[%05" PRIu64 "] [%05" PRIu64 "] max_tokens=%d, prompt=%s, is_stream=%d\n", res->session_id(),
                        res->reqwest().request_id(), max_tokens, prompt.c_str(), is_stream);

        {
            // tokenize the prompt
            auto prompt_tokens = model_general.model_string_to_tokens(prompt);

            if (prompt_tokens.size() == 0)
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "Tokenization failed - no tokens generated");

            if (prompt_tokens.size() >= max_tokens)
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "Prompt tokens size exceeds max_tokens limit");

            if (not is_stream)
            {

                // write above struct in lambda function
                std::string gen_text;
                uint32_t completion_tokens  = 0;
                uint32_t prompt_tokens_size = static_cast<uint32_t>(prompt_tokens.size());

                auto gen_text_hdl = [max_tokens, prompt_tokens_size, &gen_text,
                                     &completion_tokens](int rc, const std::string & text) -> int {
                    if (completion_tokens + prompt_tokens_size >= max_tokens or completion_tokens >= xoptions_.n_predict)
                        return -1; // end of generation
                    if (rc == 0)
                        gen_text += text;

                    completion_tokens++;
                    return 0; // continue generation
                };

                context_gen_text_until_eog(ctx, prompt_tokens, std::ref(gen_text_hdl), smpl);

#ifndef NDEBUG
                AVLLM_LOG_DEBUG("[%05" PRIu64 "] [%05" PRIu64 "] gen_text=%s\n", res->session_id(), res->reqwest().request_id(),
                                gen_text.c_str());
#endif

                json res_body = { { "id", "cmpl-" + string_generate_random(20) },
                                  { "object", "text_completion" },
                                  { "created", std::time(0) },
                                  { "model", model_name },
                                  { "system_fingerprint", "fp_44709d6fcb" },
                                  { "choices", { { { "text", gen_text }, { "index", 0 }, { "finish_reason", "length" } } } },
                                  { "usage",
                                    { { "prompt_tokens", prompt_tokens_size },
                                      { "completion_tokens", completion_tokens },
                                      { "total_tokens", prompt_tokens_size + completion_tokens } } } };

                res->set_content(res_body.dump(4));
                // res->end();
                res->endend();
            }
            else
            {

                uint32_t completion_tokens  = 0;
                uint32_t prompt_tokens_size = static_cast<uint32_t>(prompt_tokens.size());

                auto gen_text_hdl = [model_name, max_tokens, prompt_tokens_size, res,
                                     &completion_tokens](int rc, const std::string & text) {
                    if (completion_tokens + prompt_tokens_size >= max_tokens or completion_tokens >= xoptions_.n_predict)
                    {
                        res->chunk_write_async("data: " + oai_completion_chunk(model_name, "", "length"));
                        return -1; // end of generation
                    }
                    if (rc == 0)
                        res->chunk_write_async("data: " + oai_completion_chunk(model_name, text));
                    completion_tokens++;
                    return 0; // continue generation
                };

                // start writing chunk
                res->event_source_start();
                context_gen_text_until_eog(ctx, prompt_tokens, std::ref(gen_text_hdl), smpl);
                res->event_source_oai_end();
            }
        }
    };

    static auto chat_completions_handler = [&model_general](std::shared_ptr<http::response> res, int ctx_idx) -> void {
        AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(),
                                                    res->reqwest().request_id(), "chat_completions_handler")
                                  .c_str())

        llama_context * ctx       = model_general.get_context(ctx_idx);
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_sampler * smpl      = model_general.get_sampler();

        json body_ = json_parse(res->reqwest().body());
        if (body_.empty())
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "invalid request body");

        json messages_js = json_value(body_, "messages", json());
        if (messages_js.empty())
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "Missing or empty messages");

        if (body_ == json() or messages_js == json())
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Invalid JSON format");

        // ensure that the session data is initialized
        if (!res->session_data())
            res->session_data() = std::make_unique<session_chat_message>();

        std::vector<llama_chat_message> chat_messages;
        // extract promt
        for (const auto & msg : messages_js)
        {
            std::string role = (msg.contains("role") and msg.at("role").is_string()) ? msg.at("role") : "";
            if (role == "user" or role == "system" or role == "assistant" or role == "developer")
            {
                std::string content = msg.at("content").get<std::string>();
                chat_messages.push_back({ strdup(role.c_str()), strdup(content.c_str()) });
            }
        }

        std::string model_name = json_value(body_, "model", std::string("model"));
        bool is_stream         = json_value(body_, "stream", bool(false));
        json tools             = json_value(body_, "tools", json::array());

        if (is_stream)
        { // stream

            auto get_text_hdl = [res, &model_name, state = 0, cnt = 0](int rc, const std::string & text) mutable -> int {
                std::string chunk_data;
                if (rc == 0 && state == 0 && cnt++ < xoptions_.n_predict)
                {
                    chunk_data = "data: " + oai_chat_completion_chunk(model_name, text);
                    res->chunk_write_async(chunk_data);
                }
                else if ((rc < 0 || cnt >= xoptions_.n_predict) && state == 0)
                {
                    state      = 1; // end of generation
                    chunk_data = "data: " + oai_chat_completion_chunk(model_name, ".", "stop");
                    res->chunk_write_async(chunk_data);
                    return -1;
                }
                else
                    return -1;
                return 0;
            };

            std::string messages = model_oaicompact_to_text(model, body_);
            if (messages.empty())
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Failed to convert messages to text");

            AVLLM_LOG_INFO("[%05" PRIu64 "] [%05" PRIu64 "] messages=%s\n", res->session_id(), res->reqwest().request_id(),
                           messages.c_str());

            int n_tokens = -llama_tokenize(vocab, messages.data(), messages.size(), NULL, 0, true, true);

            std::vector<llama_token> prompt_tokens(n_tokens);
            llama_tokenize(vocab, messages.data(), messages.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);

            if (prompt_tokens.size() == 0)
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error,
                                         "Tokenization failed - no tokens generated");

            res->event_source_start();
            context_gen_text_until_eog(ctx, prompt_tokens, std::ref(get_text_hdl), smpl);
            res->event_source_oai_end();
        }
        else
        {
            if (tools.empty())
                AVLLM_LOG_DEBUG("[%05" PRIu64 "] [%05" PRIu64 "] no tools provided, using default empty tools\n", res->session_id(),
                                res->reqwest().request_id());
            else
                AVLLM_LOG_DEBUG("[%05" PRIu64 "] [%05" PRIu64 "] tools provided: %s\n", res->session_id(),
                                res->reqwest().request_id(), tools.dump().c_str());

            // default
            std::string content;
            uint32_t completion_tokens = 0;

            auto get_text_hdl = [&content, &completion_tokens](int rc, const std::string & text) -> int {
                if (rc == 0)
                    content += text;
                completion_tokens++;
                return 0;
            };

            std::string messages = model_oaicompact_to_text(model, body_);
            int n_tokens         = -llama_tokenize(vocab, messages.data(), messages.size(), NULL, 0, true, true);

            std::vector<llama_token> prompt_tokens(n_tokens);
            llama_tokenize(vocab, messages.data(), messages.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);

            if (prompt_tokens.size() == 0)
                HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error,
                                         "Tokenization failed - no tokens generated");

            uint32_t prompt_tokens_size = static_cast<uint32_t>(prompt_tokens.size());
            context_gen_text_until_eog(ctx, prompt_tokens, std::ref(get_text_hdl), smpl);
            json res_body = { { "id", "cmpl-" + string_generate_random(20) },
                              { "object", "text_completion" },
                              { "created", std::time(0) },
                              { "model", model_name },
                              { "system_fingerprint", "fp_44709d6fcb" },
                              { "choices", { { { "text", content }, { "index", 0 }, { "finish_reason", "length" } } } },
                              { "usage",
                                { { "prompt_tokens", prompt_tokens_size },
                                  { "completion_tokens", completion_tokens },
                                  { "total_tokens", prompt_tokens_size + completion_tokens },
                                  { "prompt_tokens_details", { { "cached_tokens", 0 }, { "audio_tokens", 0 } } },
                                  { "completion_tokens_details",
                                    { { "reasoning_tokens", 0 },
                                      { "audio_tokens", 0 },
                                      { "accepted_prediction_tokens", 0 },
                                      { "rejected_prediction_tokens", 0 } } } } },
                              { "service_tier", "default" } };

            res->set_content(res_body.dump(4));
            res->endend();
        }
    };

    auto embedding_handler = [&model_embedding](std::shared_ptr<http::response> res) -> void {
        AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(),
                                                    res->reqwest().request_id(), "embedding_handler")
                                  .c_str())

        // sanity check
        if (!model_embedding)
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Embedding model not available");
        llama_model * model = model_embedding.get();

        json body_js = json_parse(res->reqwest().body());

        if (body_js.empty())
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Invalid request content");

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
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error, "Empty input text");

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

#ifndef NDEBUG
        {
            const llama_vocab * const vocab = llama_model_get_vocab(model);
            llama_token_print(vocab, tokens);
        }
#endif // NDEBUG

        const uint64_t n_batch = cparams_emb.n_batch;
        llama_batch batch      = llama_batch_init(n_batch, 0, 1);

        for (int pos = 0; pos < tokens.size(); pos++)
            common_batch_add(batch, tokens[pos], pos, { 0 }, true);

#ifndef NDEBUG
        llama_batch_print(&batch);
#endif // NDEBUG

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
#ifndef NDEBUG
        {
            for (int i = 0; i < std::min(3, n_embd); i++)
                printf("%.6f ", *(embeddings.data() + i));
            printf("...");

            for (int i = 0; i < n_embd && i < 2; i++)
                printf("%.6f ", *(embeddings.data() + n_embd - 1 - i));
        }
#endif // NDEBUG

        json j = embeddings;
        res->set_content(j.dump(4));
        res->end();

        llama_batch_free(batch);
    };

    // infill, fim
    static auto fim_handler = [&model_general](std::shared_ptr<http::response> res, int ctx_idx) -> void {
        AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(),
                                                    res->reqwest().request_id(), "fim_handler")
                                  .c_str())
#ifdef NDEBUG
        bool silent = true;
#else
        bool silent = false;
#endif
        llama_context * ctx       = model_general.get_context(ctx_idx);
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_sampler * smpl      = model_general.get_sampler();

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

        uint32_t n_batch = llama_n_batch(ctx);
        uint32_t n_ctx   = llama_n_ctx(ctx);

        auto tokens = format_infill(vocab, input_prefix, input_suffix, body_js.at("input_extra"), n_batch, n_predict, n_ctx, false,
                                    tokenized_prompts[0]);
        if (!silent)
        {
            llama_token_print(vocab, tokens);
        }

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
            AVLLM_LOG_DEBUG("%s", "sampler:\n");
            llama_sampler_print(smpl_.get());
        }

        {
            context_gen_text_until_eog(ctx, tokens, get_text_hdl, smpl_.get());
            json body_js;
            body_js["content"] = res_body;
            res->set_content(body_js.dump());
            res->end();
            return;
        }
    };

    // health handler
    static auto health_handler = [](std::shared_ptr<http::response> res) {
        AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(),
                                                    res->reqwest().request_id(), "health_handler")
                                  .c_str())
        res->set_header("Content-Type", "application/json");
        json body;
        body["status"]  = "OK";
        body["name"]    = "av_llm";
        body["version"] = "0.0.1-Preview";
        body["uptime"]  = 0;
        res->set_content(body.dump());
        res->end();
    };

    static auto props_handler = [&model_general](std::shared_ptr<http::response> res) {
        const llama_vocab * vocab = llama_model_get_vocab(model_general.model_ptr.get());
        json data                 = {
            { "total_slots", xoptions_.n_parallel },
            { "model_path", xoptions_.model_url_or_alias },
            { "bos_token", common_token_to_piece(model_general.get_context(0), llama_vocab_bos(vocab), /* special= */ true) },
            { "eos_token", common_token_to_piece(model_general.get_context(0), llama_vocab_eos(vocab), /* special= */ true) },
        };
        res->set_content(data.dump());
        res->endend();
    };

    struct process_request_
    {
        using function_handler = std::function<void(std::shared_ptr<http::response>, int)>;
        using task             = std::tuple<function_handler, std::shared_ptr<http::response>>;

        process_request_(model_general_t & model_general_) : model_general(model_general_) {}

        void loop()
        {
            const int n_ctx = model_general.get_n_ctx();
            while (true)
            {
                {
                    std::unique_lock lk(mt);
                    cv.wait(lk, [&]() { return !tasks.empty(); });
                }
                for (int i = 0; i < n_ctx && !tasks.empty(); i++)
                {

                    llama_context * ctx = model_general.get_context(i);

                    std::unique_lock lk(mt);
                    auto func_ = std::get<0>(tasks.front());
                    auto res   = std::get<1>(tasks.front());
                    tasks.pop();
                    lk.unlock();
                    func_(res, i); // process the request
                }
            }
        }

        void operator()(std::function<void(std::shared_ptr<http::response>, int)> func_, std::shared_ptr<http::response> res_)
        {
            task task_ = std::make_tuple(func_, res_);
            {
                std::lock_guard lk(mt);
                tasks.push(task_);
                cv.notify_one(); // notify the worker to process the task
            }
        }

        model_general_t & model_general;
        std::queue<task> tasks; // queue for responses
        std::mutex mt;
        std::condition_variable cv;
    } process_request(model_general);

    auto embedding_model_handler = [&model_embedding, &embedding_handler](std::shared_ptr<http::response> res) {
        if (!model_embedding)
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::internal_server_error,
                                     "not support. the model is not inialized as request");
        embedding_handler(res);
    };

    auto oaicompact_to_text_handler = [&model_general](std::shared_ptr<http::response> res) {
        AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("[%05" PRIu64 "] [%05" PRIu64 "] %s", res->session_id(),
                                                    res->reqwest().request_id(), "context_session_oaicompact_to_text_handler")
                                  .c_str())

        nlohmann::ordered_json body_js = json_parse(res->reqwest().body());
        if (body_js.empty())
            HTTP_SEND_RES_AND_RETURN(res, http::status_code::bad_request, "invalid request body");

        std::string text = model_oaicompact_to_text(model_general.model_ptr.get(), body_js);
        res->set_content(text);
        res->endend();
    };

    http::route route_;
    // clang-format off
    // web
    route_.set_option_handler(preflight);
    route_.get("/",                      std::ref(web_handler));
    route_.get("/index.html",            std::ref(web_handler));
    // oai - model
    route_.get("/models",                std::ref(handle_models));
    route_.get("/v1/models",             std::ref(handle_models));
    route_.get("/models/{model}",        std::ref(handle_model_detail));
    route_.get("/v1/models/{model}",     std::ref(handle_model_detail));
    // add /api/tags endpoint
    route_.get("/api/tags",              std::ref(api_tags_handler));    
    route_.post("/api/show",             std::ref(api_show)); 
    route_.post("/api/chat",             [&process_request](std::shared_ptr<http::response> res) {
				process_request(std::ref(chat_completions_handler), res); 
		});
		route_.post("/v1/responses",         [&process_request](std::shared_ptr<http::response> res) {
				process_request(std::ref(responses_handler), res);
				});
    // oai - completions
    route_.post("/completions",          [&process_request](std::shared_ptr<http::response> res) {
				process_request(std::ref(completions_handler), res);
		});
    route_.post("/v1/completions",       [&process_request](std::shared_ptr<http::response> res) {
				process_request(std::ref(completions_handler), res);
		});
    // oai - chat completions
    route_.post("/chat/completions",     [&process_request](std::shared_ptr<http::response> res) {
				process_request(std::ref(chat_completions_handler), res);
		});
    route_.post("/v1/chat/completions",  [&process_request](std::shared_ptr<http::response> res) {
				process_request(std::ref(chat_completions_handler), res);
		});
		// oai - embeddings
    route_.post("/embeddings",           std::ref(embedding_model_handler));
    route_.post("/v1/embeddings",        std::ref(embedding_model_handler));
		// infill, fim (fill-in-middle)
    route_.post("/fim",                  [&process_request](std::shared_ptr<http::response> res) {
				process_request(std::ref(fim_handler), res);
		});
    route_.post("/infill",               [&process_request](std::shared_ptr<http::response> res) {
				process_request(std::ref(fim_handler), res);
		});
		// health
    route_.get("health",                 std::ref(health_handler));
		// other
		route_.post("/model/oai_to_text",    std::ref(oaicompact_to_text_handler));
		// llama.cpp
    route_.get("/props",                 std::ref(props_handler));
    // clang-format on

    AVLLM_LOG_INFO("Server can be accessed at http://127.0.0.1:%d\n", xoptions_.port);

    std::thread th(&process_request_::loop, &process_request);

    http::start_server(xoptions_.port, route_);
    th.join();
};
