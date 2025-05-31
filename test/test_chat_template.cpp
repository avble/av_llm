#include "catch2/catch.hpp"

#include "chat.h"
#include "common.h"
#include "llama.h"

#include <iostream>

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
