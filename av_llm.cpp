/* Copyright (c) 2024-2024 Harry Le (avble.harry at gmail dot com)

It can be used, modified.
*/

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include "av_connect.hpp"
#include "helper.hpp"
#include "index.html.gz.hpp"
#include "log.hpp"
#include "model.hpp"
#include "utils.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <curl/curl.h>
#include <inttypes.h>

using json = nlohmann::ordered_json;
#define MIMETYPE_JSON "application/json; charset=utf-8"

std::filesystem::path home_path;
std::filesystem::path app_path;

static void print_usage(int, char ** argv)
{
    AVLLM_LOG_INFO("%s", "\nexample usage:\n");
    AVLLM_LOG_INFO("\n    %s -m model.gguf [-p port] \n", argv[0]);
    AVLLM_LOG_INFO("%s", "\n");
}

static size_t write_data(void * ptr, size_t size, size_t nmemb, void * stream)
{
    std::ofstream * of = static_cast<std::ofstream *>(stream);

    try
    {
        of->write((const char *) ptr, size * nmemb);
        return size * nmemb;
    } catch (const std::exception & ex)
    {
        return 0;
    }
}

// Progress callback (older interface)
int progress_callback(void * /*clientp*/, curl_off_t dltotal, curl_off_t dlnow, curl_off_t /*ultotal*/, curl_off_t /*ulnow*/)
{
    if (dltotal == 0)
        return 0; // avoid division by zero

    double progress = (double) dlnow / (double) dltotal * 100.0;
    // std::cout << "\rDownload progress: " << progress << "% (" << dlnow << "/" << dltotal << " bytes)" << std::flush;
    std::cout << "\033[2K\r[";
    for (int i = 0; i < 100; i++)
        std::cout << ((i < (int) progress) ? "#" : ".");
    std::cout << "] " << (int) progress << "%" << std::flush;

    return 0; // return non-zero to abort transfer
}

// remove leading space
static auto ltrim = [](std::string & str) {
    int i = 0;
    while (i < str.size() && std::isspace(static_cast<unsigned char>(str[i])))
        i++;
    str.erase(0, i);
};

auto model_cmd_handler = [](std::string model_line) {
    std::filesystem::create_directories(app_path);

    static auto get_file_name_from_url = [](const std::string url) -> std::string {
        auto last_slash = url.find_last_of('/');

        if (last_slash == std::string::npos)
            return url;

        return url.substr(last_slash + 1);
    };

    auto model_pull = [](std::string url) {
        AVLLM_LOG_DEBUG("%s: with argument: %s \n", "model_pull", url.c_str());
        CURL * curl;
        CURLcode res;

        std::string outfilename = get_file_name_from_url(url);

        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl = curl_easy_init();
        if (curl)
        {

            auto file_path = app_path / outfilename;
            // open a file
            std::ofstream of(file_path.c_str());

            if (of.is_open())
            {
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *) &of);
                // curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

                curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

                // Set progress callback
                curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
                curl_easy_setopt(curl, CURLOPT_XFERINFODATA, nullptr);

                // follow redirect
                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

                res = curl_easy_perform(curl);
                if (res != CURLE_OK)
                    AVLLM_LOG_ERROR("curl_easy_perform() failed: %s\n", curl_easy_strerror(res));

                of.close();
            }

            AVLLM_LOG_DEBUG("%s: %d \n", "[DEBUG]", __LINE__);
            curl_easy_cleanup(curl);
        }
        else
        {
            AVLLM_LOG_ERROR("%s: %d coud not download model: %s \n", "[DEBUG]", __LINE__, url.c_str());
        }
        curl_global_cleanup();
    };

    auto model_print_header = []() {
        std::cout << std::left << std::setw(70) << "|Model path" << std::setw(1) << '|' << "Size" << '\n';
        std::cout << std::string(78, '-') << '\n'; // Increased from 58 to 78 to match new width
    };

    auto model_print = [](const std::filesystem::path & model_path, const std::uintmax_t & size) {
        std::cout << std::setw(70) << "|" + model_path.generic_string() << std::setw(1) << '|' << HumanReadable{ size } << "\n";
    };

    auto model_print_footer = []() {
        std::cout << std::string(77, '-') << std::endl; // Increased from 57 to 77 to match new width
    };

    auto model_ls = [&model_print_header, &model_print, &model_print_footer]() {
        std::vector<std::filesystem::path> search_paths = {
            app_path,                                       // ~/.av_llm
            std::filesystem::path("/usr/local/etc/.av_llm") // /usr/local/etc/.av_llm
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
                        std::filesystem::path display_path;
                        if (search_path == app_path)
                        {
                            display_path = std::filesystem::path("~/.av_llm") / entry.path().filename();
                        }
                        else
                        {
                            display_path = entry.path(); // Show full path for system directory
                        }
                        model_print(display_path, entry.file_size());
                    }
                }
            }
        }

        model_print_footer();
    };

    auto model_del = [](std::string model_name) { std::filesystem::remove(app_path / model_name); };

    std::regex model_pattern(R"((pull|del|ls)(.*))");

    std::smatch smatch_;
    if (std::regex_match(model_line, smatch_, model_pattern))
    {
        AVLLM_LOG_DEBUG("[DEBUG] %s: %d: %s\n", "model_cmd_handler", __LINE__, model_line.c_str());
        std::string sub_cmd = smatch_[1];
        // AVLLM_LOG_DEBUG("[DEBUG] %s: %d: %s\n", "model_cmd_handler", __LINE__, sub_cmd.c_str());
        std::string model_path = smatch_[2];
        // AVLLM_LOG_DEBUG("[DEBUG] %s: %d: %s \n", "model_cmd_handler", __LINE__, model_path.c_str());
        ltrim(model_path);
        // AVLLM_LOG_DEBUG("[DEBUG] %s: %d: %s\n", "model_cmd_handler", __LINE__, model_path.c_str());

        if (sub_cmd == "pull")
        {
            model_pull(model_path);
        }
        else if (sub_cmd == "del")
        {
            model_del(model_path);
        }
        else if (sub_cmd == "ls")
        {
            model_ls();
        }
        else
            AVLLM_LOG_ERROR("%s: Could not valid command: %s\n", __func__, sub_cmd.c_str());
    }
    else
        AVLLM_LOG_ERROR("%s: Wrong syntax\n", __func__);
};

auto chat_cmd_handler = [](std::string model_line) -> int {
    int n_ctx = 2048;
    std::filesystem::path model_path;

    std::regex chat_pattern(R"(([^\s]+)(.*))");

    std::smatch smatch_;

    // if (std::regex_match(model_line, smatch_, chat_pattern))
    // {
    //     if (std::filesystem::is_regular_file(std::filesystem::path(smatch_[1])))
    //         model_path = smatch_[1];
    //     else
    //         model_path = app_path / std::string(smatch_[1]);
    // }

    if (std::regex_match(model_line, smatch_, chat_pattern))
    {
        // Step 1: Check if smatch_[1] is regular .gguf file (absolute or relative to current dir)
        std::filesystem::path input_path = std::string(smatch_[1]);
        if (std::filesystem::is_regular_file(input_path) && input_path.extension() == ".gguf")
        {
            model_path = input_path;
            AVLLM_LOG_DEBUG("%s: Using model from direct path: %s\n", __func__, model_path.c_str());
        }
        // Step 2: Check in app_path (~/av_llm/)
        else if (std::filesystem::is_regular_file(app_path / input_path) && (app_path / input_path).extension() == ".gguf")
        {
            model_path = app_path / input_path;
            AVLLM_LOG_DEBUG("%s: Using model from user directory: %s\n", __func__, model_path.c_str());
        }
        // Step 3: Check in /usr/local/etc/.av_llm/
        else if (std::filesystem::is_regular_file(std::filesystem::path("/usr/local/etc/.av_llm") / input_path) &&
                 (std::filesystem::path("/usr/local/etc/.av_llm") / input_path).extension() == ".gguf")
        {
            model_path = std::filesystem::path("/usr/local/etc/.av_llm") / input_path;
            AVLLM_LOG_DEBUG("%s: Using model from system directory: %s\n", __func__, model_path.c_str());
        }
        else
        {
            AVLLM_LOG_ERROR("%s: Could not find valid .gguf model file: %s\n", __func__, input_path.c_str());
            return -1;
        }
    }
    else
    {
        AVLLM_LOG_ERROR("%s: Invalid model path format\n", __func__);
        return -1;
    }
    ggml_backend_load_all();

    // model initialized
    llama_model * model = [&model_path]() -> llama_model * {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 99;
        return llama_model_load_from_file(model_path.c_str(), model_params);
    }();
    if (model == nullptr)
    {
        AVLLM_LOG_ERROR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    // context initialize
    llama_context * ctx = [&model, &n_ctx]() -> llama_context * {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.no_perf              = false;
        ctx_params.n_ctx                = n_ctx;
        ctx_params.n_batch              = n_ctx;

        return llama_init_from_model(model, ctx_params);
    }();
    if (ctx == nullptr)
    {
        AVLLM_LOG_ERROR("%s: error: failed to create the llama_context\n", __func__);
        return -1;
    }

    // initialize the sampler
    llama_sampler * smpl = []() {
        auto sparams    = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        auto smpl       = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        return smpl;
    }();
    if (smpl == nullptr)
    {
        AVLLM_LOG_ERROR("%s: error: could not create sampling\n", __func__);
        return -1;
    }

    const char * chat_tmpl = llama_model_chat_template(model, /* name */ nullptr);
    if (chat_tmpl == nullptr)
    {
        AVLLM_LOG_ERROR("%s: error: could no accept the template is null\n", __func__);
        return -1;
    }
    std::vector<llama_chat_message> chat_messages;
    std::vector<char> chat_message_output(llama_n_ctx(ctx));
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
                return 1;
            }

            chat_message_end = len;
        }
        std::string prompt(chat_message_output.begin() + chat_message_start, chat_message_output.begin() + chat_message_end);

        chat_message_start = chat_message_end;

        llama_token new_token;
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr)
        {
            AVLLM_LOG_ERROR("%s: failed to get vocal from model \n", __func__);
            exit(-1);
        }

        bool is_first       = llama_kv_self_used_cells(ctx) == 0;
        int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);

        if (llama_tokenize(vocab, prompt.data(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0)
        {
            AVLLM_LOG_ERROR("%s: failed to tokenize the prompt \n", __func__);
            exit(-1);
        }

        if (false)
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
                    exit(-1);
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
            int n_ctx      = llama_n_ctx(ctx);
            int n_ctx_used = llama_kv_self_used_cells(ctx);

            if (n_ctx_used + batch.n_tokens > n_ctx)
            {
                AVLLM_LOG_WARN("%s: the context is exceeded. \n", __func__);
                return -1;
            }

            if (llama_decode(ctx, batch))
            {
                AVLLM_LOG_ERROR("%s : failed to eval, return code %d\n", __func__, 1);
                return -1;
            }

            new_token = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, new_token))
            {
                break;
            }

            char buf[100];
            int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                AVLLM_LOG_ERROR("%s, failed to convert a token \n", __func__);
                return 0;
            }

            std::string out(buf, n);
            AVLLM_LOG("%s", out.c_str());
            fflush(stdout);

            batch = llama_batch_get_one(&new_token, 1);
        }
    }

    if (false)
    {
        llama_perf_sampler_print(smpl);
        llama_perf_context_print(ctx);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
};

auto server_cmd_handler = [](std::string run_line) -> int { // run serve
    std::string model_path;
    int srv_port = 8080;

    std::regex chat_pattern(R"(([^\s]+)(.*))");

    std::smatch smatch_;

    if (std::regex_match(run_line, smatch_, chat_pattern))
    {
        model_path = smatch_[1];
    }
    if (model_path == "")
        return -1;

    ggml_backend_load_all();

    // model initialized
    llama_model * model = [&model_path]() -> llama_model * {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 99;
        return llama_model_load_from_file(model_path.c_str(), model_params);
    }();
    if (model == nullptr)
    {
        AVLLM_LOG_ERROR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    // // context initialize
    auto me_llama_context_init = [&model]() -> llama_context * {
        int n_ctx                       = 2048;
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.no_perf              = false;
        ctx_params.n_ctx                = n_ctx;
        ctx_params.n_batch              = n_ctx;

        return llama_init_from_model(model, ctx_params);
    };

    // chat template
    auto chat_templates = common_chat_templates_init(model, "");

    try
    {
        common_chat_format_example(chat_templates.get(), false);
    } catch (const std::exception & e)
    {
        AVLLM_LOG_WARN("%s: Chat template parsing error: %s\n", __func__, e.what());
        AVLLM_LOG_WARN(
            "%s: The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the "
            "model to output suboptimal responses\n",
            __func__);
        chat_templates = common_chat_templates_init(model, "chatml");
    }
    {
        AVLLM_LOG_INFO("%s: chat template: %s \n", __func__, common_chat_templates_source(chat_templates.get()));
        AVLLM_LOG_INFO("%s: chat example: %s \n", __func__, common_chat_format_example(chat_templates.get(), false).c_str());
    }

    // initialize the sampler
    llama_sampler * smpl = []() {
        auto sparams    = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        auto smpl       = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        return smpl;
    }();
    if (smpl == nullptr)
    {
        AVLLM_LOG_ERROR("%s: error: could not create sampling\n", __func__);
        return -1;
    }

    struct chat_session_t
    {
        llama_context * ctx;
        std::vector<char> chat_message_output;
        int chat_message_start;
        int chat_message_end;

        const bool add_generation_prompt = true;

        chat_session_t(llama_context * _ctx)
        {
            chat_message_start = 0;
            chat_message_end   = 0;
            ctx                = _ctx;
            chat_message_output.resize(llama_n_ctx(ctx));
            // chat_tmpl = llama_model_chat_template(model, /* name */ nullptr);
        }

        ~chat_session_t()
        {
            AVLLM_LOG_TRACE_SCOPE("~chat_session_t");
            if (ctx != nullptr)
            {
                llama_free(ctx);
                AVLLM_LOG_DEBUG("%s\n", "~chat_seesion_t is ended.");
            }
            ctx = nullptr;
        }
    };

    auto chat_sesion_get = [&chat_templates, &model, &smpl](chat_session_t & chat,
                                                            const std::vector<llama_chat_message> & chat_messages,
                                                            std::function<void(int, std::string)> func_) -> int {
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
                func_(-1, "");
                return -1;
            }

            chat.chat_message_end = len;
        }
        std::string prompt(chat.chat_message_output.begin() + chat.chat_message_start,
                           chat.chat_message_output.begin() + chat.chat_message_end);

        chat.chat_message_start = chat.chat_message_end;

        llama_token new_token;
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr)
        {
            AVLLM_LOG_ERROR("%s: failed to get vocal from model \n", __func__);
            func_(-1, "");
            return -1;
        }

        int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);

        if (llama_tokenize(vocab, prompt.data(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
        {
            AVLLM_LOG_ERROR("%s: failed to tokenize the prompt \n", __func__);
            func_(-1, "");
            return -1;
        }

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        while (true)
        {
            int n_ctx      = llama_n_ctx(chat.ctx);
            int n_ctx_used = llama_kv_self_used_cells(chat.ctx);

            if (n_ctx_used + batch.n_tokens > n_ctx)
            {
                AVLLM_LOG_WARN("%s: the context is exceeded. \n", __func__);
                func_(-1, "");
                return -1;
            }

            if (llama_decode(chat.ctx, batch))
            {
                AVLLM_LOG_ERROR("%s : failed to eval, return code %d\n", __func__, 1);
                func_(-1, "");
                return -1;
            }

            new_token = llama_sampler_sample(smpl, chat.ctx, -1);
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
            AVLLM_LOG_INFO("%s", out.c_str());
            // fflush(stdout);

            func_(0, out);

            batch = llama_batch_get_one(&new_token, 1);
        }

        AVLLM_LOG_INFO("%s", "\n");
        return 0;
    };

    static auto completions_chat_handler = [&](http::response res) -> void {
        AVLLM_LOG_TRACE_SCOPE("completions_chat_handler")
        AVLLM_LOG_DEBUG("%s: session-id: %" PRIu64 "\n", "completions_chat_handler", res.session_id());

        enum CHAT_TYPE
        {
            CHAT_TYPE_DEFAULT = 0,
            CHAT_TYPE_STREAM  = 1
        };

        CHAT_TYPE chat_type = CHAT_TYPE_DEFAULT;

        json body_;
        json messages_js;
        [&body_, &messages_js](const std::string & body) mutable {
            try
            {
                body_       = json::parse(body);
                messages_js = body_.at("messages");
            } catch (const std::exception & ex)
            {
                body_       = {};
                messages_js = {};
            }
        }(res.reqwest().body());

        if (body_ == json() or messages_js == json())
        {
            res.result() = http::status_code::internal_error;
            res.end();
            return;
        }

        if (body_.contains("stream") and body_.at("stream").is_boolean() and body_["stream"].get<bool>() == true)
        {
            chat_type = CHAT_TYPE_STREAM;
        }

        if (chat_type == CHAT_TYPE_DEFAULT or chat_type == CHAT_TYPE_STREAM)
        {
            try
            {
                if (not res.get_data().has_value())
                {
                    AVLLM_LOG_INFO("%s: initialize context for session id: %" PRIu64 " \n", "completions_chat_handler",
                                   res.session_id());
                    llama_context * p = me_llama_context_init();
                    if (p != nullptr)
                        res.get_data().emplace<chat_session_t>(p);
                    else
                        throw std::runtime_error("can not initalize context");
                }
            } catch (const std::exception & e)
            {
                AVLLM_LOG_WARN("%s: can not initialize the context \n", __func__);
                res.result() = http::status_code::internal_error;
                res.end();
                return;
            }

            chat_session_t & chat_session = std::any_cast<chat_session_t &>(res.get_data());

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
            {
                std::string res_body;
                auto get_text_hdl = [&](int rc, const std::string & text) -> int {
                    if (rc == 0)
                        res_body = res_body + text;

                    return 0;
                };
                chat_sesion_get(chat_session, chat_messages, get_text_hdl);

                res.set_content(res_body);
                res.end();
            }
            else
            {
                res.event_source_start();
                auto get_text_hdl = [&](int rc, const std::string & text) -> int {
                    if (rc == 0)
                        res.chunk_write("data: " + oai_make_stream(text));

                    return 0;
                };

                chat_sesion_get(chat_session, chat_messages, get_text_hdl);

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                res.chunk_end();
            }
        }
        else
        {
            std::cout << "should handle the end of chunk message" << std::endl;
            res.result() = http::status_code::internal_error;
            res.end();
        }
    };

    static auto handle_models = [&](http::response res) {
        AVLLM_LOG_TRACE_SCOPE("handle_models")
        json models = { { "object", "list" },
                        { "data",
                          {
                              { { "id", "openchat_3.6" },
                                { "object", "model" },
                                { "created", std::time(0) },
                                { "owned_by", "avbl llm" },
                                { "meta", "meta" } },
                          } } };

        res.set_content(models.dump(), MIMETYPE_JSON);
        res.end();
    };

    struct handle_static_file
    {
        handle_static_file(std::string _file_path, std::string _content_type)
        {
            file_path    = _file_path;
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

    static auto preflight = [](http::response res) {
        res.set_header("Access-Control-Allow-Origin", res.reqwest().get_header("origin"));
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_content("", "text/html");
        res.end();
    };

    http::route route_;

    route_.set_option_handler(preflight);

    static auto web_handler = [&](http::response res) -> void {
        if (res.reqwest().get_header("accept-encoding").find("gzip") == std::string::npos)
        {
            res.set_content("Error: gzip is not supported by this browser", "text/plain");
        }
        else
        {
            res.set_header("Content-Encoding", "gzip");
            // COEP and COOP headers, required by pyodide (python interpreter)
            res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
            res.set_header("Cross-Origin-Opener-Policy", "same-origin");
            res.set_content(reinterpret_cast<const char *>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
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
    route_.post("/completions", [](http::response res) { std::thread{ completions_chat_handler, std::move(res) }.detach(); });
    route_.post("/v1/completions", [](http::response res) { std::thread{ completions_chat_handler, std::move(res) }.detach(); });
    // chat
    route_.post("/chat/completions", [](http::response res) { std::thread{ completions_chat_handler, std::move(res) }.detach(); });
    route_.post("/v1/chat/completions",
                [](http::response res) { std::thread{ completions_chat_handler, std::move(res) }.detach(); });

    AVLLM_LOG_INFO("Server is started at http://127.0.0.1:%d\n", srv_port);
    http::start_server(srv_port, route_);

    LOG("\n");

    llama_model_free(model);
    return 0;
};

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

    app_path = home_path / ".av_llm";

    for (int i = 0; i < argc; i++)
        AVLLM_LOG_DEBUG("[%03d]:%s\n", i, argv[i]);

    if (argc == 1)
    {
        AVLLM_LOG_ERROR("%s\n", "error systax. Please refer to manual");
        exit(0);
    }

    pre_config_model_init();

    auto chat_cmd_caller = [](std::string model_description) -> int { // handle the syntax
        // $./av_llm <model-description>
        // which is alias to $./av_llm chat

        {
            // handle if the argv[1] is *.guff
            std::filesystem::path model_filename = model_description;
            if (model_filename.extension() == ".gguf")
            {

                std::optional<std::filesystem::path> model_path = std::filesystem::is_regular_file(model_filename) ? model_filename
                    : std::filesystem::is_regular_file(app_path / model_filename)
                    ? std::optional<std::filesystem::path>(app_path / model_filename)
                    : std::nullopt;

                if (!model_path.has_value())
                    chat_cmd_handler(model_path.value());

                return 0;
            }
        }

        {
            // handle precofig model
            if (pre_config_model.find(model_description) != pre_config_model.end())
            {
                std::string url = pre_config_model[model_description];

                AVLLM_LOG_DEBUG("%s:%d model url: %s\n", __func__, __LINE__, url.c_str());

                auto last_slash = url.find_last_of("/");
                if (last_slash != std::string::npos)
                {
                    std::filesystem::path model_filename = url.substr(last_slash + 1);

                    AVLLM_LOG_DEBUG("%s:%d model path: %s\n", __func__, __LINE__, model_filename.c_str());

                    std::optional<std::filesystem::path> model_path = std::filesystem::is_regular_file(app_path / model_filename)
                        ? std::optional<std::filesystem::path>(app_path / model_filename)
                        : std::nullopt;

                    if (!model_path.has_value())
                    {
                        AVLLM_LOG_DEBUG("%s:%d does NOT have model. So download it: %s from url: %s \n", __func__, __LINE__,
                                        model_filename.generic_string().c_str(), url.c_str());
                        model_cmd_handler("pull " + url);
                    }
                    else
                        AVLLM_LOG_DEBUG("%s:%d have model at:  %s \n", __func__, __LINE__,
                                        model_path.value().generic_string().c_str());

                    chat_cmd_handler((app_path / model_filename).generic_string());
                }
                return 0;
            }
        }
        { // todo: handle the descrition (http://)
        }

        return 0;
    };

    // print each argument
    std::string line;
    for (int i = 1; i < argc; i++)
        line += ((i == 1 ? "" : " ") + std::string(argv[i]));

    {
        const std::regex pattern(R"((serve|chat|model)(.*))");
        std::smatch match;
        if (!std::regex_match(line, match, pattern))
            chat_cmd_caller(argv[1]);
    }

    {
        const std::regex pattern(R"((serve|chat|model)(.*))");
        std::smatch match;
        if (std::regex_match(line, match, pattern))
        {
            AVLLM_LOG_DEBUG("[DEBUG] %s:%d:%s\n", __func__, __LINE__, line.c_str());
            std::string command;
            command = match[1];

            if (command == "model")
            {
                std::string model_cmd = match[2];
                AVLLM_LOG_DEBUG("[DEBUG] %s:%d:%s\n", __func__, __LINE__, model_cmd.c_str());
                ltrim(model_cmd);
                model_cmd_handler(model_cmd);
            }
            else if (command == "chat")
            {
                std::string model_desc = match[2];
                ltrim(model_desc);
                // chat_cmd_handler(cmd);
                AVLLM_LOG_DEBUG("%s:%d - chat with model-description: %s\n", __func__, __LINE__, model_desc.c_str());
                chat_cmd_caller(model_desc);
            }
            else if (command == "serve")
            {
                // check if argument is *.gguf
                {
                    AVLLM_LOG_DEBUG("%s:%d\n", __func__, __LINE__);
                    std::string arg = match[2];
                    ltrim(arg);
                    std::filesystem::path path = arg;
                    if (path.extension() == ".gguf")
                    {
                        AVLLM_LOG_DEBUG("%s:%d\n", __func__, __LINE__);
                        if (std::filesystem::is_regular_file(path)) // or std::filesystem::is_regular_file(app_path / path)))
                        {
                            server_cmd_handler(path.generic_string());
                        }
                        else if (std::filesystem::is_regular_file(app_path / path))
                        {
                            server_cmd_handler((app_path / path).generic_string());
                        }
                        return 0;
                    }
                }

                // check if preconfigured model
                {
                    std::string arg = match[2];
                    ltrim(arg);
                    AVLLM_LOG_DEBUG("%s:%d info: %s \n", __func__, __LINE__, arg.c_str());
                    if (pre_config_model.find(arg) != pre_config_model.end())
                    {
                        AVLLM_LOG_DEBUG("%s:%d\n", __func__, __LINE__);
                        std::string url = pre_config_model[arg];

                        AVLLM_LOG_DEBUG("%s:%d model path: %s\n", __func__, __LINE__, url.c_str());

                        auto last_slash = url.find_last_of("/");
                        if (last_slash != std::string::npos)
                        {
                            std::filesystem::path model_filename = url.substr(last_slash + 1);

                            AVLLM_LOG_DEBUG("%s:%d model path: %s\n", __func__, __LINE__, model_filename.c_str());

                            std::optional<std::filesystem::path> model_path = std::filesystem::is_regular_file(model_filename)
                                ? model_filename
                                : std::filesystem::is_regular_file(app_path / model_filename)
                                ? std::optional<std::filesystem::path>(app_path / model_filename)
                                : std::nullopt;

                            if (!model_path.has_value())
                                model_cmd_handler("pull " + url);

                            server_cmd_handler((app_path / model_filename).generic_string());
                        }
                        return 0;
                    }
                }

                {
                    AVLLM_LOG_ERROR("%s:%d \n", "args", __LINE__);
                }
            }
            else
                AVLLM_LOG_ERROR("%s:%d \n", "args", __LINE__);
        }

        return 0;
    }
}
