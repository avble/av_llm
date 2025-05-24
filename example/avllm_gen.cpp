/*
This sample tool is insprised from llama.cpp (simple)
*/

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <cstring>

#include <inttypes.h>

static void print_usage(int, char ** argv)
{
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf -p prompt\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv)
{

    std::string model_path;
    std::string prompt;
    [&argc, &argv](auto & model_path, auto & prompt) { // parsing the argument
        int i = 0;
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i], "-m") == 0)
            {
                if (i + 1 < argc)
                {
                    model_path = argv[++i];
                }
                else
                {
                    print_usage(1, argv);
                }
            }
            else if (strcmp(argv[i], "-p") == 0)
            {
                if (i + 1 < argc)
                {
                    prompt = argv[++i];
                }
                else
                {
                    print_usage(1, argv);
                }
            }
        }
    }(model_path, prompt);
    if (model_path == "" or prompt == "")
    {
        print_usage(1, argv);
        return 1;
    }
    std::cout << "prompt: " << prompt << std::endl;

    ggml_backend_load_all();

    // model initialized
    llama_model * model = [model_path]() -> llama_model * {
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
    llama_context * ctx = [&model, &prompt]() -> llama_context * {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_prompt        = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.no_perf              = false;
        ctx_params.n_ctx                = 2048;
        ctx_params.n_batch              = 2048;

        return llama_init_from_model(model, ctx_params);
    }();
    if (ctx == nullptr)
    {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return -1;
    }

    // initialize the sampler
    llama_sampler * smpl = []() {
        auto sparams         = llama_sampler_chain_default_params();
        sparams.no_perf      = false;
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        return smpl;
    }();
    if (smpl == nullptr)
    {
        fprintf(stderr, "%s: error: could not create sampling\n", __func__);
        return 1;
    }

    // decode the prompt
    {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr)
        {
            fprintf(stderr, "%s: failed to get vocal from model \n", __func__);
            return -1;
        }

        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        // std::cout << n_prompt_tokens << std::endl;
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);

        if (llama_tokenize(vocab, prompt.data(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
        {
            fprintf(stderr, "%s: failed to tokenize the prompt \n", __func__);
            return -1;
        }

        if (true)
        { // print token
            std::cout << "tokens: \n";
            for (auto token : prompt_tokens)
            {
                char buf[120];
                int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
                if (n < 0)
                {
                    fprintf(stderr, "%s: error: failed to tokenize \n", __func__);
                    return 1;
                }
                std::string s(buf, n);
                std::cout << s;
            }
            std::cout << "\n";
        }

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return -1;
        }
    }

    // do inference
    {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const auto t_main_start   = ggml_time_us();
        llama_token new_token_id;
        int num_token = 0;
        while (num_token++ < 100)
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id))
                break;

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                break;
            }
            std::string text(buf, n);
            printf("%s", text.c_str());
            fflush(stdout);

            llama_batch batch = llama_batch_get_one(&new_token_id, 1);
            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch))
            {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return -1;
            }
        }
    }

    if (true)
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
