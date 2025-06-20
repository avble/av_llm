#include "catch2/catch.hpp"

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

#ifdef _MSC_VER
#include <ciso646>
#endif

TEST_CASE("test_chat_template")
{
    if (false)
    { // test: <llama.cpp>/src/llama.cpp
        std::vector<llama_chat_message> conversation{
            { "system", "You are a helpful assistant" },
            { "user", "Hello" },
            { "assistant", "Hi there" },
            { "user", "Who are you" },
            { "assistant", "   I am an assistant   " },
            { "user", "Another question" },
        };

        const auto add_generation_prompt = false;

        std::string template_str =
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ "
            "raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if "
            "message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' "
            "%}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') "
            "}}{% endif %}{% endfor %}";

        std::string expected_output = "[INST] You are a helpful assistant\nHello [/INST]Hi there</s>[INST] Who are you [/INST]   I "
                                      "am an assistant   </s>[INST] Another question [/INST]";

        std::string formatted_chat;
        formatted_chat.resize(1024);
        auto res = llama_chat_apply_template(template_str.c_str(), conversation.data(), conversation.size(), add_generation_prompt,
                                             formatted_chat.data(), formatted_chat.size());

        formatted_chat.resize(res);
        std::cout << formatted_chat << std::endl;
        REQUIRE(formatted_chat == expected_output);
    }

    { // test: <llama.cpp>/src/llama-chat.cpp

        std::vector<llama_chat_message> conversation{
            { "system", "You are a helpful assistant" },
            { "user", "Hello" },
            { "assistant", "Hi there" },
            { "user", "Who are you" },
            { "assistant", "   I am an assistant   " },
            { "user", "Another question" },
        };

        std::string template_str =
            "{% for message in messages %}{% set role = message['role'] | lower %}{% if role == 'user' %}{% set role = 'HUMAN' "
            "%}{% endif %}{% set role = role | upper %}{{ '<role>' + role + '</role>' + message['content'] }}{% endfor %}{% if "
            "add_generation_prompt %}{{ '<role>ASSISTANT</role>' }}{% endif %}";

        std::string expected = "<role>SYSTEM</role>You are a helpful assistant<role>HUMAN</role>Hello<role>ASSISTANT</role>Hi "
                               "there<role>HUMAN</role>Who are you<role>ASSISTANT</role>   I am an assistant   "
                               "<role>HUMAN</role>Another question<role>ASSISTANT</role>";

        std::string bos_token = "";
        std::string eos_token = "";

        auto simple_msg = [](const std::string & role, const std::string & content) -> common_chat_msg {
            common_chat_msg msg;
            msg.role    = role;
            msg.content = content;
            return msg;
        };

        std::vector<common_chat_msg> messages;
        for (const auto & msg : conversation)
            messages.push_back(simple_msg(msg.role, msg.content));

        bool add_generation_prompt = true;

        common_chat_templates_inputs inputs;
        inputs.use_jinja             = false;
        inputs.messages              = messages;
        inputs.add_generation_prompt = add_generation_prompt;

        auto tmpls = common_chat_templates_init(/* model= */ nullptr, template_str.c_str(), bos_token, eos_token);
        {
            std::cout << "chat template: \n" << common_chat_templates_source(tmpls.get()) << std::endl;
            std::cout << "example of chat message: \n"
                      << common_chat_format_example(tmpls.get(), add_generation_prompt).c_str() << std::endl;
        }

        auto output = common_chat_templates_apply(tmpls.get(), inputs).prompt;
        REQUIRE(expected == output);
    }

    {
    }
}

TEST_CASE("test")
{
#ifdef _WIN32
    char * env = std::getenv("USERPROFILE");
    if (env == nullptr)
        std::cout << "null" << std::endl;
    else
        std::cout << std::string(env) << std::endl;
#endif
}

TEST_CASE("test_fim_model")
{

    // test preparation
    // fim model in under current working directory
    common_params params;
    {
        // config
    }
    std::filesystem::path model_path("../../../model/qwen2.5-coder-3b-instruct-q8_0.gguf");

    if (not std::filesystem::is_regular_file(model_path))
    {
        INFO(model_path.generic_string() + " is not existed");
        REQUIRE(false);
    }

    INFO("step 01");
    ggml_backend_load_all();

    INFO("step: model init");
    llama_model * model = [&model_path]() -> llama_model * {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 20;
        return llama_model_load_from_file(model_path.generic_string().c_str(), model_params);
    }();
    if (nullptr == model)
    {
        INFO("fail at loading model");
        REQUIRE(false);
    }

    INFO("step: model context init");

    llama_context * model_ctx = [&model]() -> llama_context * {
        // llama_context_params ctx_params = llama_context_params();
        llama_context_params ctx_params = llama_context_default_params();
        return llama_init_from_model(model, ctx_params);
    }();
    if (nullptr == model_ctx)
    {
        INFO("failed at initializing context");
        REQUIRE(false);
    }

    INFO("step: sampling");

    llama_sampler * smpl = []() {
        auto smpl_params     = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(smpl_params);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        return smpl;
    }();
    if (nullptr == smpl)
    {
        INFO("failed at initialing the sampler");
        REQUIRE(false);
    }
    {
        // verbose print sampling chain
        printf("Sampling: \n");
        [&smpl]() {
            int n_samplers = llama_sampler_chain_n(smpl);
            for (int i = 0; i < n_samplers; i++)
            {
                llama_sampler * smpl_ = llama_sampler_chain_get(smpl, i);
                printf("[%d][%s] -> ", i, llama_sampler_name(smpl_));
            }
        }();
        printf("\n");
    }

    {

        INFO("step: decoding prompt");
        std::string prompt = R"(<|fim_prefix|>def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    <|fim_suffix|>
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)<|fim_middle|>)";

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
        {
            INFO("failed to tokenize the prompt \n");
            REQUIRE(false);
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
                    INFO("fail tokenize \n");
                    REQUIRE(false);
                }
                std::string s(buf, n);
                std::cout << s;
            }
            std::cout << "\n";
        }

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(model_ctx, batch))
        {
            INFO("failed to eval, return code \n");
            REQUIRE(false);
        }

        INFO("step: do inference");
        std::cout << "Inference: \n" << std::endl;
        int num_decoded_token = 0;
        llama_token new_token_id;
        while (num_decoded_token < 1000)
        {
            new_token_id = llama_sampler_sample(smpl, model_ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id))
            {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                INFO("failed at token_to_piece\n");
                REQUIRE(false);
            }
            std::string text(buf, n);
            std::cout << text;
            fflush(stdout);

            llama_batch batch = llama_batch_get_one(&new_token_id, 1);
            if (llama_decode(model_ctx, batch))
            {
                INFO("failed at llama_decode");
                REQUIRE(false);
            }
            num_decoded_token++;
        }

        std::cout << std::endl;
    }
}

TEST_CASE("llam_cli_01")
{
    common_params params;
    common_params_context ctx_arg(params);

#if 0
    ctx_arg.options.push_back(common_arg({ "--model-01" }, "play around with model 01", [](common_param & params) {
        params.model.hf_repo = "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF";
        params.model.hf_file = "qwen2.5-coder-3b-q8_0.gguf";
        params.port          = 8012;
        params.n_gpu_layers  = 99;
        params.flash_attn    = true;
        params.n_ubatch      = 1024;
        params.n_batch       = 1024;
        params.n_ctx         = 0;
        params.n_cache_reuse = 256;
    }));
#endif
}
