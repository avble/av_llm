#include "catch2/catch.hpp"

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <thread>
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
    {
        /*
         * given: conversation & jinja template
         * output: format the chat [to feed to the model]
         * notes:
         * - the jinja template is provided as a hint to detect the type (model)
         * - then the function llm_chat_apply_template do a simple formatting
         */
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
        int res = llama_chat_apply_template(template_str.c_str(), conversation.data(), conversation.size(), add_generation_prompt,
                                            formatted_chat.data(), formatted_chat.size());

        if (res > formatted_chat.size())
        {
            formatted_chat.resize(res);
            llama_chat_apply_template(template_str.c_str(), conversation.data(), conversation.size(), add_generation_prompt,
                                      formatted_chat.data(), formatted_chat.size());
        }

        std::cout << formatted_chat << std::endl;
        REQUIRE(formatted_chat == expected_output);
    }

    if (false)
    {
        /*
         * given: conversation and jinja template
         * output: the formated of chat conversion
         * Notes:
         * - the jinja can be obtain from model or provided text
         * - use_jinja: is to determine if use the minja engine to render text or others
         *
         */

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

        auto tmpls = common_chat_templates_init(/* model= */ nullptr, template_str.c_str(), bos_token, eos_token);
        {
            std::cout << "chat template: \n" << common_chat_templates_source(tmpls.get()) << std::endl;
            std::cout << "example of chat message: \n"
                      << common_chat_format_example(tmpls.get(), add_generation_prompt).c_str() << std::endl;
        }

        {
            // use the legacy to format the text
            common_chat_templates_inputs inputs;
            inputs.use_jinja             = false;
            inputs.messages              = messages;
            inputs.add_generation_prompt = add_generation_prompt;

            auto output = common_chat_templates_apply(tmpls.get(), inputs).prompt;
            REQUIRE(expected == output);
        }
        {
            // use the minja to format the text
            common_chat_templates_inputs inputs;
            inputs.use_jinja             = true;
            inputs.messages              = messages;
            inputs.add_generation_prompt = add_generation_prompt;

            auto output = common_chat_templates_apply(tmpls.get(), inputs).prompt;
            REQUIRE(expected == output);
        }
    }

    if (true)
    {
        /*
         * given: jinja and oai input
         * output: the format chat
         * notes:
         * -
         */

        // Qwen3-8B template
        std::string qwen3_8b_tmpl = R"( {%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for index in range(ns.last_query_index, -1, -1) %}
    {%- set message = messages[index] %}
    {%- if ns.multi_step_tool and message.role == "user" and not('<tool_response>' in message.content and '</tool_response>' in message.content) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %})";

        std::string qwen_25_7b_tmpl = R"(
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
)";

        std::string oai_input_str_1 = R"(
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that can use tools to get information for the user."
    },
    {
      "role": "user",
      "content": "What's the weather like in New York?"
    }
  ],
  "add_generation_prompt": true
}
)";

        std::string oai_input_str_2 = R"(
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that can use tools to get information for the user."
    },
    {
      "role": "user",
      "content": "What's the weather like in New York?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": [
                "celsius",
                "fahrenheit"
              ],
              "description": "The unit of temperature to use"
            }
          },
          "required": [
            "location"
          ]
        }
      }
    }
  ],
  "add_generation_prompt": true
}
)";
        // std::string & template_jinja = qwen_25_7b_tmpl;
        std::string & template_jinja = qwen3_8b_tmpl;
        std::string & oai_str        = oai_input_str_2;

        json oai_js      = json::parse(oai_str);
        json messages_js = oai_js.at("messages");

        std::string bos_token = "";
        std::string eos_token = "";

        std::vector<common_chat_msg> messages = common_chat_msgs_parse_oaicompat(messages_js);

        json tools_js                       = oai_js.at("tools");
        std::vector<common_chat_tool> tools = common_chat_tools_parse_oaicompat(tools_js);

        const auto add_generation_prompt = true;

        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.messages              = messages;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.tools                 = tools;

        auto tmpls  = common_chat_templates_init(/* model= */ nullptr, template_jinja.c_str(), bos_token, eos_token);
        auto output = common_chat_templates_apply(tmpls.get(), inputs).prompt;

        std::cout << "\n>>>>>>>>>>>>\n";
        std::cout << output;
        std::cout << "\n<<<<<<<<<<<<\n";
    }
}

int context_gen_text_until_eog(llama_context * ctx, std::vector<llama_token> & prompt_tokens, llama_sampler * smpl)
{
    llama_token new_token;
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    int count = 1;

    while (count++ < 10)
    {
        int n_ctx      = llama_n_ctx(ctx);
        int n_ctx_used = llama_kv_self_used_cells(ctx);

        if (n_ctx_used + batch.n_tokens > n_ctx)
        {
            printf("%s: the context is exceeded. \n", __func__);
            return -1;
        }

        if (llama_decode(ctx, batch))
        {
            printf("%s : failed to eval, return code %d\n", __func__, 1);
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
            printf("%s, failed to convert a token \n", __func__);
            return -1;
        }

        std::string out(buf, n);
        std::cout << out;

        batch = llama_batch_get_one(&new_token, 1);
    }

    return 0; // return -1 means end of the ctx_session_ session
};

TEST_CASE("test_context_01")
{

    ggml_backend_load_all();
    llama_model_params model_param = llama_model_default_params();
    model_param.n_gpu_layers       = 99;
    std::string str                = "hello world";

    llama_model * model       = llama_load_model_from_file("../../../model/qwen2.5-coder-3b-instruct-q8_0.gguf", model_param);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_token               = -llama_tokenize(vocab, str.data(), str.size(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_token);
    llama_tokenize(vocab, str.data(), str.size(), tokens.data(), tokens.size(), true, true);

    for (int i = 0; i < 30; i++)
    {
        llama_context_params ctx_param = llama_context_default_params();
        llama_context * ctx            = llama_init_from_model(model, ctx_param);
        llama_context_ptr ctx_ptr(ctx);

        auto sparams          = llama_sampler_chain_default_params();
        sparams.no_perf       = false;
        llama_sampler * smpl_ = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());
        auto sampler_default_ptr = llama_sampler_ptr(smpl_);

        context_gen_text_until_eog(ctx_ptr.get(), tokens, sampler_default_ptr.get());
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // llama_free(ctx);
    }

    llama_model_free(model);
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
