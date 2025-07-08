/*
given a chatML format, generate the tokens

i.e a chatML messages
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "what is the meaning of life?"
  }
]
*/

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string.h>
#include <thread>
#include <vector>

#include <inttypes.h>

#ifdef _MSC_VER
#include <ciso646>
#endif

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

template <typename T>
static T json_value(const json & js, const std::string & key, const T & default_value)
{
    // Fallback null to default value
    if (js.contains(key) && !js.at(key).is_null())
    {
        try
        {
            return js.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &)
        {
            printf("Wrong type supplied for parameter '%s'. Expected '%s', using default value\n", key.c_str(),
                   json(default_value).type_name());
            return default_value;
        }
    }
    else
    {
        return default_value;
    }
}

static void print_usage(int, char ** argv)
{
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-c ctx]\n", argv[0]);
    printf("Options: \n"
           "-c: number of context\n"
           "-ngl: number of GPU layer"
           "-input: text | @file");
    printf("\n");
}

std::string model_path;
int n_ctx = 2048;
int n_gl  = 0;
std::string messages;
int main(int argc, char ** argv)
{

    [&argc, &argv](auto & model_path, auto & n_ctx) { // parsing the argument
        int i = 0;
        try
        {
            for (int i = 1; i < argc; i++)
            {
                if (strcmp(argv[i], "-m") == 0)
                {
                    if (i + 1 < argc)
                        model_path = argv[++i];
                    else
                        print_usage(1, argv);
                }
                else if (strcmp(argv[i], "-c") == 0)
                {
                    if (i + 1 < argc)
                        n_ctx = std::stoi(argv[++i]);
                    else
                        print_usage(1, argv);
                }
                else if (strcmp(argv[i], "-ngl") == 0)
                {
                    if (i + 1 < argc)
                        n_gl = std::stoi(argv[++i]);
                    else
                        print_usage(1, argv);
                }
                else if (strcmp(argv[i], "-input") == 0)
                {
                    if (i + 1 < argc)
                        messages = argv[++i];
                    else
                        print_usage(1, argv);
                }
            }
        } catch (const std::exception & ex)
        {
            fprintf(stdout, "%s:%d , exception: %s \n", __func__, __LINE__, ex.what());
        }
    }(model_path, n_ctx);
    if (model_path.empty() or messages.empty())
    {
        print_usage(1, argv);
        return 1;
    }

    if (messages.size() > 1 and messages[0] == '@')
    {
        std::fstream f_in(messages.substr(1));
        if (not f_in.is_open())
        {
            std::cerr << "Could not open file " << messages.substr(1) << std::endl;
            exit(-1);
        }
        std::stringstream ss;
        ss << f_in.rdbuf();
        messages = ss.str();
    }

    ggml_backend_load_all();

    // model initialized
    llama_model * model = []() -> llama_model * {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 99;
        return llama_model_load_from_file(model_path.c_str(), model_params);
    }();
    if (model == nullptr)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // context initialize
    llama_context * ctx = [&model]() -> llama_context * {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.no_perf              = false;
        ctx_params.n_ctx                = n_ctx;
        ctx_params.n_batch              = n_ctx;

        return llama_init_from_model(model, ctx_params);
    }();
    if (ctx == nullptr)
    {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
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
        fprintf(stderr, "%s: error: could not create sampling\n", __func__);
        return -1;
    }

    const char * chat_tmpl = llama_model_chat_template(model, /* name */ nullptr);
    if (chat_tmpl == nullptr)
    {
        fprintf(stderr, "%s: error: could no accept the template is null\n", __func__);
        return -1;
    }
    std::vector<char> chat_message_output(llama_n_ctx(ctx));
    int chat_message_size = 0;

    // convert chatML format to llama_chat_message
    try
    {
        json messages_js = json::parse(messages);
        for (const auto & msg : messages_js)
        {
            std::string role = json_value(msg, "role", std::string());
            std::string text = json_value(msg, "content", std::string());
            if (role.empty() or text.empty())
                continue;

            const llama_chat_message chat_message({ role.c_str(), text.c_str() });

            int len = llama_chat_apply_template(chat_tmpl, &chat_message, 1, true, chat_message_output.data() + chat_message_size,
                                                chat_message_output.size() - chat_message_size);
            if (len > chat_message_size)
            {
                chat_message_output.resize(chat_message_output.size() + len);
                len = llama_chat_apply_template(chat_tmpl, &chat_message, 1, true, chat_message_output.data() + chat_message_size,
                                                chat_message_output.size() - chat_message_size);
            }

            if (len < 0)
            {
                fprintf(stderr, "%s: error: failed to apply chat template", __func__);
                exit(-1);
            }
            chat_message_size += len;
        }

    } catch (const json::exception & ex)
    {
        std::cerr << "err: parsing messages" << std::endl;
        exit(1);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == nullptr)
    {
        fprintf(stderr, "%s: failed to get vocal from model \n", __func__);
        exit(-1);
    }

    bool is_first       = llama_kv_self_used_cells(ctx) == 0;
    int n_prompt_tokens = -llama_tokenize(vocab, &chat_message_output[0], chat_message_size, NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);

    if (llama_tokenize(vocab, &chat_message_output[0], chat_message_size, prompt_tokens.data(), prompt_tokens.size(), is_first,
                       true) < 0)
    {
        fprintf(stderr, "%s: failed to tokenize the prompt \n", __func__);
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
                fprintf(stderr, "%s: error: failed to tokenize \n", __func__);
                exit(-1);
            }
            std::string s(buf, n);
            std::cout << s;
            cnt++;
        }
        std::cout << "end: " << cnt << "\n";
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    llama_token new_token;
    while (true)
    {
        int n_ctx      = llama_n_ctx(ctx);
        int n_ctx_used = llama_kv_self_used_cells(ctx);

        if (n_ctx_used + batch.n_tokens > n_ctx)
        {
            fprintf(stdout, "%s: the context is exceeded. \n", __func__);
            exit(-1);
        }

        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            exit(-1);
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
            fprintf(stderr, "%s, failed to convert a token \n", __func__);
            exit(0);
        }

        std::string out(buf, n);
        printf("%s", out.c_str());
        fflush(stdout);

        batch = llama_batch_get_one(&new_token, 1);
    }

    if (false)
    {
        printf("\n");
        llama_perf_sampler_print(smpl);
        llama_perf_context_print(ctx);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
