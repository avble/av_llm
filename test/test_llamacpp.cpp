#include "catch2/catch.hpp"

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _MSC_VER
#include <ciso646>
#endif

using namespace std;

TEST_CASE("llama_arg_parser")
{

    // task: list all options associated with llama_example
    if (true)
    {
        common_params common_param;
        auto llama_example_to_str = [](llama_example ex) -> std::string {
            switch (ex)
            {
            case LLAMA_EXAMPLE_COMMON:
                return "LLAMA_EXAMPLE_COMMON";
            case LLAMA_EXAMPLE_SPECULATIVE:
                return "LLAMA_EXAMPLE_SPECULATIVE";
            case LLAMA_EXAMPLE_MAIN:
                return "LLAMA_EXAMPLE_MAIN";
            case LLAMA_EXAMPLE_EMBEDDING:
                return "LLAMA_EXAMPLE_EMBEDDING";
            case LLAMA_EXAMPLE_PERPLEXITY:
                return "LLAMA_EXAMPLE_PERPLEXITY";
            case LLAMA_EXAMPLE_RETRIEVAL:
                return "LLAMA_EXAMPLE_RETRIEVAL";
            case LLAMA_EXAMPLE_PASSKEY:
                return "LLAMA_EXAMPLE_PASSKEY";
            case LLAMA_EXAMPLE_IMATRIX:
                return "LLAMA_EXAMPLE_IMATRIX";
            case LLAMA_EXAMPLE_BENCH:
                return "LLAMA_EXAMPLE_BENCH";
            case LLAMA_EXAMPLE_SERVER:
                return "LLAMA_EXAMPLE_SERVER";
            case LLAMA_EXAMPLE_CVECTOR_GENERATOR:
                return "LLAMA_EXAMPLE_CVECTOR_GENERATOR";
            case LLAMA_EXAMPLE_EXPORT_LORA:
                return "LLAMA_EXAMPLE_EXPORT_LORA";
            case LLAMA_EXAMPLE_MTMD:
                return "LLAMA_EXAMPLE_MTMD";
            case LLAMA_EXAMPLE_LOOKUP:
                return "LLAMA_EXAMPLE_LOOKUP";
            case LLAMA_EXAMPLE_PARALLEL:
                return "LLAMA_EXAMPLE_PARALLEL";
            case LLAMA_EXAMPLE_TTS:
                return "LLAMA_EXAMPLE_TTS";
            case LLAMA_EXAMPLE_COUNT:
                return "LLAMA_EXAMPLE_COUNT";
            default:
                return "UNKNOWN";
            }
        };

        // prepare data
        std::map<std::string, int> data; /* = { { "row-1", { "item 1", "item 2", "", "item 4" } },
                                                                 { "row-2", { "item 1", "", "item 3", "item 4" } } }; */

        for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++)
        {

            // std::cout << "\x1b[31m" << llama_example_to_str(static_cast<llama_example>(ex)).c_str() << "\x1b[0m\n";
            auto ctx_arg = common_params_parser_init(common_param, (enum llama_example) ex);
            {
                // print all option of ctx arg
                for (const auto & opt : ctx_arg.options)
                    for (const auto & arg : opt.args)
                    {
                        // change to green color
                        // std::cout << "\x1b[32m" << "\t" << arg << "\x1b[0m" << std::endl;
                        data[arg] = data[arg] | (1 << static_cast<int>(ex));
                    }
            }
        }

        // Helper: center text in fixed width
        static auto centerText = [](const string & text, int width) {
            int padding = width - text.size();
            int left    = padding / 2;
            int right   = padding - left;
            return string(left, ' ') + text + string(right, ' ');
        };

        // Helper: print row separator line
        // void printSeparator(int columns, int cell_width)

        data["--alias"] = 7;
        if (true)
        {
            int row_count    = data.size();
            int column_count = 0;
            if (!data.empty())
                column_count = static_cast<int>(LLAMA_EXAMPLE_COUNT);

            // string, remove the prefix
            auto remove_prefix_LLAMA_EXAMPLE = [](std::string str) -> std::string {
                if (str.find("LLAMA_EXAMPLE_") == 0)
                    str.erase(0, 12);
                return str;
            };

            std::array<int, LLAMA_EXAMPLE_COUNT> cell_widths;
            cell_widths[0] = 20;
            for (int i = 0; i < LLAMA_EXAMPLE_COUNT; i++)
            {
                std::string str    = llama_example_to_str(static_cast<llama_example>(i));
                str                = remove_prefix_LLAMA_EXAMPLE(str);
                cell_widths[i + 1] = str.size();
            }

            static auto printSeparator = [&cell_widths](int columns) {
                cout << "|";
                for (int i = 0; i < columns + 1; ++i)
                { // +1 for row header
                    cout << string(cell_widths[i], '-') << "|";
                }
                cout << endl;
            };
            // Print header (first row)
            cout << "|" << centerText("", cell_widths[0]);
            for (int i = 0; i < LLAMA_EXAMPLE_COUNT; i++)
            {
                std::string str = llama_example_to_str(static_cast<llama_example>(i));
                str             = remove_prefix_LLAMA_EXAMPLE(str);
                cout << "|" << centerText(str, cell_widths[i + 1]);
            }
            cout << "|\n";

            printSeparator(column_count);

            // For each column, print the column name and v's from each row
            for (auto row : data)
            {
                // cout << row.first << std::endl;
                // cout << "|" << centerText(row.first, cell_widths[0]);
                //
                cout << "|" << centerText(row.first, cell_widths[0]);
                for (int col = 0; col < static_cast<int>(LLAMA_EXAMPLE_COUNT); col++)
                {
                    string cell = ((row.second & (1 << col)) != 0) ? "v" : "";
                    cout << "|" << centerText(cell, cell_widths[col + 1]);
                }
                cout << "|\n";
                printSeparator(column_count);
            }
        }
    }

    if (false)
    { // test: given an argment, check the return common param
        common_params params;
        // lambda function, convert from vector of string to vectory of char *
        auto vector_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
            std::vector<char *> res;
            for (auto & arg : argv)
            {
                res.push_back(const_cast<char *>(arg.c_str()));
            }
            return res;
        };

        // print all memmber of common_params
        auto print_all_llama_example = [](const common_params & params) {
            std::cout << "common_params:" << std::endl;
            std::cout << "  port: " << params.port << std::endl;
            std::cout << "  n_predict: " << params.n_predict << std::endl;
            std::cout << "  n_ctx: " << params.n_ctx << std::endl;
            std::cout << "  n_batch: " << params.n_batch << std::endl;
            std::cout << "  n_ubatch: " << params.n_ubatch << std::endl;
            std::cout << "  n_keep: " << params.n_keep << std::endl;
            // printf the all member of model struct
            std::cout << "  param.model.path: " << params.model.path << std::endl;
            std::cout << "  param.model.url: " << params.model.url << std::endl;
            std::cout << "  param.model.hf_repo: " << params.model.hf_repo << std::endl;
            std::cout << "  param.model.hf_file: " << params.model.hf_file << std::endl;
        };

        printf("before: \n");
        print_all_llama_example(params);

        std::vector<std::string> args = { "cli_app", "--fim-qwen-3b-default" };
        if (!common_params_parse(args.size(), vector_to_char(args).data(), params, LLAMA_EXAMPLE_SPECULATIVE))
        {
            printf("failed \n");
            REQUIRE(false);
        }

        printf("after: \n");
        print_all_llama_example(params);
    }
}

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
            "message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == "
            "'assistant' "
            "%}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are "
            "supported!') "
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

        std::string template_str = "{% for message in messages %}{% set role = message['role'] | lower %}{% if role == "
                                   "'user' %}{% set role = 'HUMAN' "
                                   "%}{% endif %}{% set role = role | upper %}{{ '<role>' + role + '</role>' + "
                                   "message['content'] }}{% endfor %}{% if "
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

    const llama_vocab * vocab = llama_model_get_vocab(model);

    INFO("step: sampling");
    llama_sampler * smpl = [&vocab]() {
        auto smpl_params     = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(smpl_params);
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.89, 20));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(4294967295));
        // llama_sampler_chain_add(smpl, llama_sampler_init_infill(vocab));
        // llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
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
#define TC 2
#if TC == 1
        std::string prompt = R"(<|fim_prefix|>def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    <|fim_suffix|>
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)<|fim_middle|>)";
#endif
#if TC == 2
        // test repeation
        std::string prompt = R"(<|fim_prefix|>#include <iostream>
int main(int argc, char * argv[])
{
    printf("hello world \n");<|fim_suffix|>

    return 0;
}<|fim_middle|>)";

#endif
        // printf("[DEBUG] %s: %d \n", __func__, __LINE__);
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

        // printf("[DEBUG] %s: %d \n", __func__, __LINE__);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(model_ctx, batch))
        {
            INFO("failed to eval, return code \n");
            REQUIRE(false);
        }

        printf("[DEBUG] %s: %d \n", __func__, __LINE__);

        INFO("step: do inference");
        std::cout << "Inference: \n" << std::endl;
        int num_decoded_token = 0;
        llama_token new_token_id;
        while (num_decoded_token < 1000)
        {
            // new_token_id = llama_sampler_sample(smpl, model_ctx, -1);
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
