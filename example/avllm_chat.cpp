/*
Generate the tokens based on the input


- prompt template
openAI compatible format
{
  "messages": [
    {
      "role": "user",
      "content": "this is user"
    },
    {
      "role": "user",
      "content": "this is user"
    }
  ]
}

- user input
[text] --> [chatML] --> [
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "what is the meaning of life?"
    }
  ]
}

*/

#include <inttypes.h>
#include <string.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#ifdef _MSC_VER
#include <ciso646>
#endif

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

class null_buffer : public std::streambuf {
 public:
  int overflow(int c) override { return c; }
};

class null_stream : public std::ostream {
 public:
  null_stream() : std::ostream(&null_buffer_) {}

 private:
  null_buffer null_buffer_;
};

#ifdef NDEBUG
static null_stream av_trace;
#else
static std::ostream &av_trace = std::cout;
#endif

#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_RESET "\033[0m"

template <typename T>
static T json_value(const json &js, const std::string &key,
                    const T &default_value) {
  // Fallback null to default value
  if (js.contains(key) && !js.at(key).is_null()) {
    try {
      return js.at(key);
    } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
      printf(
          "Wrong type supplied for parameter '%s'. Expected '%s', using "
          "default value\n",
          key.c_str(), json(default_value).type_name());
      return default_value;
    }
  } else {
    return default_value;
  }
}

static void print_usage(int, char **argv) {
  printf("\nexample usage:\n");
  printf("\n    %s [Options] -m model.gguf\n", argv[0]);
  printf(
      "Options: \n"
      "-i,--interactive:      enable the interactive mode. More detail, see "
      "description\n"
      "\n"  // model
      "-ngl:                  number of GPU layer. More detail, see "
      "description\n"
      "--jinja:               use jinja or not. More detail, see description\n"
      "--chat-template-file   chat template from file\n"
      "--jinja-file:          jinja file from input. if not set, it is from "
      "model. More detail, see description\n"
      "\n"  // context
      "-c:                    number of context. More detail, see description\n"
      "");
  printf("\n");
}

common_params cparams;
// default

int main(int argc, char **argv) {

  try {
    // default
    cparams.n_ctx = 0;
    cparams.use_jinja = true;
    cparams.interactive = true;
    common_params_parse(argc, argv, cparams, LLAMA_EXAMPLE_MAIN, print_usage);

    if (cparams.model.path.empty()) {
      print_usage(1, argv);
      return 1;
    }

    if (cparams.prompt.size() > 1 and cparams.prompt[0] == '@') {
      std::fstream f_in(cparams.prompt.substr(1));
      if (not f_in.is_open()) {
        std::cerr << "Could not open file " << cparams.prompt.substr(1)
                  << std::endl;
        exit(-1);
      }
      std::stringstream ss;
      ss << f_in.rdbuf();
      cparams.prompt = ss.str();
    }

    ggml_backend_load_all();

    // model initialized
    llama_model *model = []() -> llama_model * {
      llama_model_params model_params = llama_model_default_params();
      return llama_model_load_from_file(cparams.model.path.c_str(),
                                        model_params);
    }();
    if (model == nullptr) {
      fprintf(stderr, "%s: error: unable to load model\n", __func__);
      return 1;
    }

    // context initialize
    llama_context *ctx = [&model]() -> llama_context * {
      llama_context_params ctx_params = llama_context_default_params();
      ctx_params.no_perf = false;
      ctx_params.n_ctx = cparams.n_ctx;
      ctx_params.n_batch = cparams.n_ctx;

      return llama_init_from_model(model, ctx_params);
    }();
    if (ctx == nullptr) {
      fprintf(stderr, "%s: error: failed to create the llama_context\n",
              __func__);
      return -1;
    }

    // initialize the sampler
    llama_sampler *smpl = []() {
      auto sparams = llama_sampler_chain_default_params();
      sparams.no_perf = false;
      auto smpl = llama_sampler_chain_init(sparams);
      llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
      return smpl;
    }();
    if (smpl == nullptr) {
      fprintf(stderr, "%s: error: could not create sampling\n", __func__);
      return -1;
    }

    const char *chat_tmpl =
        llama_model_chat_template(model, /* name */ nullptr);
    if (chat_tmpl == nullptr) {
      fprintf(stderr, "%s: error: could no accept the template is null\n",
              __func__);
      return -1;
    }

    json oai_js;
    try {
      oai_js = json::parse(cparams.prompt);
    } catch (const json::exception &ex) {
      std::cerr << "exception: " << ex.what() << std::endl;
    }

    if (cparams.interactive) {
      // std::string & template_jinja = qwen_25_7b_tmpl;
      std::string &template_jinja = cparams.chat_template;

      printf("[DEBUG] template_jinja: %s\n", template_jinja.c_str());

      // parsing the oai input
      std::string bos_token = "";
      std::string eos_token = "";
      auto tmpls = common_chat_templates_init(model, template_jinja.c_str(),
                                              bos_token, eos_token);
      bool add_generation_prompt =
          oai_js.contains("add_generation_prompt")
              ? oai_js.at("add_generation_prompt").get<bool>()
              : false;

      std::vector<common_chat_msg> common_chat_messages =
          common_chat_msgs_parse_oaicompat(oai_js.at("messages"));

      auto oai_js_to_model_text = [&model, &template_jinja,
                                   &tmpls](const json &oai_js) {
        std::vector<common_chat_msg> messages =
            common_chat_msgs_parse_oaicompat(oai_js.at("messages"));
        std::vector<common_chat_tool> tools =
            oai_js.contains("tools")
                ? common_chat_tools_parse_oaicompat(oai_js.at("tools"))
                : std::vector<common_chat_tool>();

        bool add_generation_prompt =
            oai_js.contains("add_generation_prompt")
                ? oai_js.at("add_generation_prompt").get<bool>()
                : false;

        common_chat_templates_inputs inputs;
        inputs.messages = messages;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.tools = tools;
        inputs.use_jinja = cparams.use_jinja;
        return common_chat_templates_apply(tmpls.get(), inputs).prompt;
      };

      auto tokenize = [&ctx, &model](const std::string &text) {
        const llama_vocab *vocab = llama_model_get_vocab(model);
        bool is_first =
            llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
        int n_prompt_tokens = -llama_tokenize(vocab, &text[0], text.size(),
                                              NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, &text[0], text.size(), prompt_tokens.data(),
                           prompt_tokens.size(), is_first, true) < 0) {
          fprintf(stderr, "%s: failed to tokenize the prompt \n", __func__);
          exit(-1);
        }

        if (true) {  // print token
          av_trace << "\ntokens: \n";
          int cnt = 0;
          for (auto token : prompt_tokens) {
            char buf[120];
            int n =
                llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
            if (n < 0) {
              fprintf(stderr, "%s: error: failed to tokenize \n", __func__);
              exit(-1);
            }
            std::string s(buf, n);
            av_trace << s;
            cnt++;
          }
          av_trace << "token[end]: " << cnt << "\n";
        }

        return prompt_tokens;
      };

      auto gen_text_until_eog = [&model, &ctx, &smpl](llama_batch &batch) {
        llama_token new_token;
        const llama_vocab *vocab = llama_model_get_vocab(model);
        std::cout << COLOR_GREEN;
        do {
          int n_ctx = llama_n_ctx(ctx);
          int n_ctx_used =
              llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

          if (n_ctx_used + batch.n_tokens > n_ctx) {
            fprintf(stdout, "%s: the context is exceeded. \n", __func__);
            exit(-1);
          }

          new_token = llama_sampler_sample(smpl, ctx, -1);
          if (llama_vocab_is_eog(vocab, new_token)) {
            break;
          }

          char buf[100];
          int n =
              llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
          if (n < 0) {
            fprintf(stderr, "%s, failed to convert a token \n", __func__);
            exit(0);
          }

          std::string out(buf, n);
          printf("%s", out.c_str());
          fflush(stdout);

          batch = llama_batch_get_one(&new_token, 1);
          if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__,
                    1);
            exit(-1);
          }

        } while (true);
        std::cout << COLOR_RESET;
      };

      auto model_prompt_text = oai_js_to_model_text(oai_js);

      auto prompt_tokens = tokenize(model_prompt_text);
      llama_batch batch =
          llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

      if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        exit(-1);
      }

      if (add_generation_prompt) gen_text_until_eog(batch);

      // interactive decode
      while (true) {
        std::string line;
        std::cout << "\n[avllm_chat] >";
        std::getline(std::cin, line);
        if (line.empty()) continue;

        json oai_input_js_;
        if (line[0] == '@') {
          std::fstream if_in(line.substr(1));

          if (!if_in.is_open()) {
            std::cerr << "error open file\n";
          }

          std::stringstream ss;
          ss << if_in.rdbuf();
          oai_input_js_ = json::parse(ss.str());
        } else {
          json j_a = json::array();
          j_a.push_back({{"role", "user"}, {"content", line}});
          oai_input_js_ = {{"messages", j_a}};
        }

        std::string msg_str;
        std::vector<common_chat_msg> common_chat_msg__ =
            common_chat_msgs_parse_oaicompat(oai_input_js_.at("messages"));

        for (auto it = common_chat_msg__.begin(); it != common_chat_msg__.end();
             it++) {
          bool add_ass = std::next(it) == common_chat_msg__.end();
          msg_str += common_chat_format_single(
              tmpls.get(), common_chat_messages, *it, add_ass, true);
        }

        av_trace << "msg: next\n";
        av_trace << msg_str << std::endl;
        av_trace << "msg: next[end]\n";

        auto prompt_tokens = tokenize(msg_str);
        llama_batch batch_ =
            llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(ctx, batch_)) {
          fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
          exit(-1);
        }

        gen_text_until_eog(batch_);
      }

      return 0;
    }

    std::vector<char> chat_message_output(llama_n_ctx(ctx));
    int chat_message_size = 0;

    // convert chatML format to llama_chat_message
    try {
      json messages_js = oai_js.at("messages");
      for (const auto &msg : messages_js) {
        std::string role = json_value(msg, "role", std::string());
        std::string text = json_value(msg, "content", std::string());
        if (role.empty() or text.empty()) continue;

        const llama_chat_message chat_message({role.c_str(), text.c_str()});

        int len = llama_chat_apply_template(
            chat_tmpl, &chat_message, 1, true,
            chat_message_output.data() + chat_message_size,
            chat_message_output.size() - chat_message_size);
        if (len > chat_message_size) {
          chat_message_output.resize(chat_message_output.size() + len);
          len = llama_chat_apply_template(
              chat_tmpl, &chat_message, 1, true,
              chat_message_output.data() + chat_message_size,
              chat_message_output.size() - chat_message_size);
        }

        if (len < 0) {
          fprintf(stderr, "%s: error: failed to apply chat template", __func__);
          exit(-1);
        }
        chat_message_size += len;
      }

    } catch (const json::exception &ex) {
      std::cerr << "err: parsing messages" << std::endl;
      exit(1);
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (vocab == nullptr) {
      fprintf(stderr, "%s: failed to get vocal from model \n", __func__);
      exit(-1);
    }

    bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
    int n_prompt_tokens =
        -llama_tokenize(vocab, &chat_message_output[0], chat_message_size, NULL,
                        0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);

    if (llama_tokenize(vocab, &chat_message_output[0], chat_message_size,
                       prompt_tokens.data(), prompt_tokens.size(), is_first,
                       true) < 0) {
      fprintf(stderr, "%s: failed to tokenize the prompt \n", __func__);
      exit(-1);
    }

    if (false) {  // print token
      av_trace << "\ntokens: \n";
      int cnt = 0;
      for (auto token : prompt_tokens) {
        char buf[120];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
          fprintf(stderr, "%s: error: failed to tokenize \n", __func__);
          exit(-1);
        }
        std::string s(buf, n);
        av_trace << s;
        cnt++;
      }
      av_trace << "end: " << cnt << "\n";
    }

    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    llama_token new_token;
    while (true) {
      int n_ctx = llama_n_ctx(ctx);
      int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

      if (n_ctx_used + batch.n_tokens > n_ctx) {
        fprintf(stdout, "%s: the context is exceeded. \n", __func__);
        exit(-1);
      }

      if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        exit(-1);
      }

      new_token = llama_sampler_sample(smpl, ctx, -1);
      if (llama_vocab_is_eog(vocab, new_token)) {
        break;
      }

      char buf[100];
      int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
      if (n < 0) {
        fprintf(stderr, "%s, failed to convert a token \n", __func__);
        exit(0);
      }

      std::string out(buf, n);
      std::cout << COLOR_RED << out;
      std::flush(std::cout);

      batch = llama_batch_get_one(&new_token, 1);
    }

    if (false) {
      printf("\n");
      llama_perf_sampler_print(smpl);
      llama_perf_context_print(ctx);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

  } catch (const std::exception &ex) {
    std::cerr << "Exception: " << ex.what() << std::endl;
  }

  return 0;
}

/*
 * use-case: function calling
 * $
 *
 *
 *
 */
