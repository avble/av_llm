/* Copyright (c) 2024-2024 Harry Le (avble.harry at gmail dot com)

It can be used, modified.
*/

#include <inttypes.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#include "../log.hpp"
#include "./index.html.gz.hpp"
#include "arg.h"
#include "av_connect.hpp"
#include "chat.h"
#include "common.h"
#include "helper.hpp"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#ifdef _MSC_VER
#include <ciso646>
#endif

using json = nlohmann::ordered_json;
#define MIMETYPE_JSON "application/json; charset=utf-8"

static void print_usage(int, char **argv) {
  AVLLM_LOG_INFO("%s", "\nexample usage:\n");
  AVLLM_LOG_INFO("\n    %s -m model.gguf [-p port] \n", argv[0]);
  AVLLM_LOG_INFO("%s", "\n");
}

int main(int argc, char **argv) {
  std::string model_path;
  int srv_port = 8080;

  {  // parsing the argument
    try {
      int i = 0;
      for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
          if (i + 1 < argc) {
            model_path = argv[++i];
          } else {
            print_usage(1, argv);
            return 1;
          }
        } else if (strcmp(argv[i], "-p") == 0) {
          if (i + 1 < argc) {
            srv_port = std::stoi(argv[++i]);
          }
        }
      }
    } catch (const std::exception &ex) {
      AVLLM_LOG_DEBUG("%s", ex.what());
      print_usage(1, argv);
      exit(1);
    }
  }

  if (model_path == "") {
    print_usage(1, argv);
    return 1;
  }

  ggml_backend_load_all();

  // model initialized
  llama_model *model = [&model_path]() -> llama_model * {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    return llama_model_load_from_file(model_path.c_str(), model_params);
  }();
  if (model == nullptr) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return 1;
  }

  // // context initialize
  auto me_llama_context_init = [&model]() -> llama_context * {
    int n_ctx = 2048;
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.no_perf = false;
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    return llama_init_from_model(model, ctx_params);
  };

  // chat template
  auto chat_templates = common_chat_templates_init(model, "");

  try {
    common_chat_format_example(chat_templates.get(), false);
  } catch (const std::exception &e) {
    AVLLM_LOG_WARN("%s: Chat template parsing error: %s\n", __func__, e.what());
    AVLLM_LOG_WARN(
        "%s: The chat template that comes with this model is not yet "
        "supported, falling back to chatml. This may cause the "
        "model to output suboptimal responses\n",
        __func__);
    chat_templates = common_chat_templates_init(model, "chatml");
  }
  {
    AVLLM_LOG_INFO("%s: chat template: %s \n", __func__,
                   common_chat_templates_source(chat_templates.get()));
    AVLLM_LOG_INFO(
        "%s: chat example: %s \n", __func__,
        common_chat_format_example(chat_templates.get(), false).c_str());
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

  struct chat_session_t {
    llama_context *ctx;
    std::vector<char> chat_message_output;
    int chat_message_start;
    int chat_message_end;

    const bool add_generation_prompt = true;

    chat_session_t(llama_context *_ctx) {
      chat_message_start = 0;
      chat_message_end = 0;
      ctx = _ctx;
      chat_message_output.resize(llama_n_ctx(ctx));
      // chat_tmpl = llama_model_chat_template(model, /* name */ nullptr);
    }

    ~chat_session_t() {
      AVLLM_LOG_TRACE_SCOPE("~chat_session_t");
      if (ctx != nullptr) {
        llama_free(ctx);
        AVLLM_LOG_DEBUG("%s\n", "~chat_seesion_t is ended.");
      }
      ctx = nullptr;
    }
  };

  auto chat_sesion_get =
      [&chat_templates, &model, &smpl](
          chat_session_t &chat,
          const std::vector<llama_chat_message> &chat_messages,
          std::function<void(int, std::string)> func_) -> int {
    {  // get input string and apply template

      auto chat_tmpl = common_chat_templates_source(chat_templates.get());

      int len = llama_chat_apply_template(
          chat_tmpl, chat_messages.data(), chat_messages.size(), true,
          chat.chat_message_output.data(), chat.chat_message_output.size());

      if (len > (int)chat.chat_message_output.size()) {
        chat.chat_message_output.resize(len);
        len = llama_chat_apply_template(
            chat_tmpl, chat_messages.data(), chat_messages.size(), true,
            chat.chat_message_output.data(), chat.chat_message_output.size());
      }

      if (len < 0) {
        fprintf(stderr, "%s: error: failed to apply chat template", __func__);
        func_(-1, "");
        return -1;
      }

      chat.chat_message_end = len;
    }
    std::string prompt(
        chat.chat_message_output.begin() + chat.chat_message_start,
        chat.chat_message_output.begin() + chat.chat_message_end);

    chat.chat_message_start = chat.chat_message_end;

    llama_token new_token;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (vocab == nullptr) {
      fprintf(stderr, "%s: failed to get vocal from model \n", __func__);
      func_(-1, "");
      return -1;
    }

    int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                          NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);

    if (llama_tokenize(vocab, prompt.data(), prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), true,
                       true) < 0) {
      fprintf(stderr, "%s: failed to tokenize the prompt \n", __func__);
      func_(-1, "");
      return -1;
    }

    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    while (true) {
      int n_ctx = llama_n_ctx(chat.ctx);
      int n_ctx_used = llama_kv_self_used_cells(chat.ctx);

      if (n_ctx_used + batch.n_tokens > n_ctx) {
        fprintf(stdout, "%s: the context is exceeded. \n", __func__);
        func_(-1, "");
        return -1;
      }

      if (llama_decode(chat.ctx, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        func_(-1, "");
        return -1;
      }

      new_token = llama_sampler_sample(smpl, chat.ctx, -1);
      if (llama_vocab_is_eog(vocab, new_token)) {
        break;
      }

      char buf[100];
      int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
      if (n < 0) {
        fprintf(stderr, "%s, failed to convert a token \n", __func__);
        func_(-1, "");
        return -1;
      }

      std::string out(buf, n);
      printf("%s", out.c_str());
      fflush(stdout);

      func_(0, out);

      batch = llama_batch_get_one(&new_token, 1);
    }

    AVLLM_LOG_INFO("%s", "\n");
    return 0;
  };

  static auto completions_chat_handler = [&](http::response res) -> void {
    AVLLM_LOG_TRACE_SCOPE("completions_chat_handler")
    AVLLM_LOG_DEBUG("%s: session-id: %" PRIu64 "\n", "completions_chat_handler",
                    res.session_id());

    enum CHAT_TYPE { CHAT_TYPE_DEFAULT = 0, CHAT_TYPE_STREAM = 1 };

    CHAT_TYPE chat_type = CHAT_TYPE_DEFAULT;

    json body_;
    json messages_js;
    [&body_, &messages_js](const std::string &body) mutable {
      try {
        body_ = json::parse(body);
        messages_js = body_.at("messages");
      } catch (const std::exception &ex) {
        body_ = {};
        messages_js = {};
      }
    }(res.reqwest().body());

    if (body_ == json() or messages_js == json()) {
      res.result() = http::status_code::internal_server_error;
      res.end();
      return;
    }

    if (body_.contains("stream") and body_.at("stream").is_boolean() and
        body_["stream"].get<bool>() == true) {
      chat_type = CHAT_TYPE_STREAM;
    }

    if (chat_type == CHAT_TYPE_DEFAULT or chat_type == CHAT_TYPE_STREAM) {
      try {
        if (not res.get_data().has_value()) {
          AVLLM_LOG_INFO("%s: initialize context for session id: %" PRIu64
                         " \n",
                         "completions_chat_handler", res.session_id());
          llama_context *p = me_llama_context_init();
          if (p != nullptr)
            res.get_data().emplace<chat_session_t>(p);
          else
            throw std::runtime_error("can not initalize context");
        }
      } catch (const std::exception &e) {
        AVLLM_LOG_WARN("%s: can not initialize the context \n", __func__);
        res.result() = http::status_code::internal_server_error;
        res.end();
        return;
      }

      chat_session_t &chat_session =
          std::any_cast<chat_session_t &>(res.get_data());

      std::vector<llama_chat_message> chat_messages;
      std::string prompt;

      // extract promt
      for (const auto &msg : messages_js) {
        std::string role = (msg.contains("role") and msg.at("role").is_string())
                               ? msg.at("role")
                               : "";
        if (role == "user" or role == "system" or role == "assistant") {
          std::string content = msg.at("content").get<std::string>();
          chat_messages.push_back(
              {strdup(role.c_str()), strdup(content.c_str())});
          std::cout << role << ":" << content << std::endl;
        }
      }

      if (chat_type == CHAT_TYPE_DEFAULT) {
        std::string res_body;
        auto get_text_hdl = [&](int rc, const std::string &text) -> int {
          if (rc == 0) res_body = res_body + text;

          return 0;
        };
        chat_sesion_get(chat_session, chat_messages, get_text_hdl);

        res.set_content(res_body);
        res.end();
      } else {
        res.event_source_start();
        auto get_text_hdl = [&](int rc, const std::string &text) -> int {
          if (rc == 0) res.chunk_write("data: " + oai_make_stream(text));

          return 0;
        };

        chat_sesion_get(chat_session, chat_messages, get_text_hdl);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        res.chunk_end();
      }
    } else {
      std::cout << "should handle the end of chunk message" << std::endl;
      res.result() = http::status_code::internal_error;
      res.end();
    }
  };

  static auto handle_models = [&](http::response res) {
    AVLLM_LOG_TRACE_SCOPE("handle_models")
    json models = {{"object", "list"},
                   {"data",
                    {
                        {{"id", "openchat_3.6"},
                         {"object", "model"},
                         {"created", std::time(0)},
                         {"owned_by", "avbl llm"},
                         {"meta", "meta"}},
                    }}};

    res.set_content(models.dump(), MIMETYPE_JSON);
    res.end();
  };

  struct handle_static_file {
    handle_static_file(std::string _file_path, std::string _content_type) {
      file_path = _file_path;
      content_type = _content_type;
    }
    void operator()(http::response res) {
      AVLLM_LOG_TRACE_SCOPE("handle_static_file")
      if (not std::filesystem::exists(std::filesystem::path(file_path))) {
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
    res.set_header("Access-Control-Allow-Origin",
                   res.reqwest().get_header("origin"));
    res.set_header("Access-Control-Allow-Credentials", "true");
    res.set_header("Access-Control-Allow-Methods", "POST");
    res.set_header("Access-Control-Allow-Headers", "*");
    res.set_content("", "text/html");
    res.end();
  };

  http::route route_;

  route_.set_option_handler(preflight);

  // Web (yeap, I known it currently it reads from file each time!)
  // route_.get("/", handle_static_file("simplechat/index.html", "text/html;
  // charset=utf-8")); route_.get("/index.html",
  // handle_static_file("simplechat/index.html", "text/html; charset=utf-8"));
  // route_.get("/datautils.mjs", handle_static_file("simplechat/datautils.mjs",
  // "text/javascript; charset=utf-8")); route_.get("/simplechat.css",
  // handle_static_file("simplechat/simplechat.css", "text/css;
  // charset=utf-8")); route_.get("/simplechat.js",
  // handle_static_file("simplechat/simplechat.js", "text/javascript"));
  // route_.get("/ui.mjs", handle_static_file("simplechat/ui.mjs",
  // "text/javascript; charset=utf-8"));

  static auto web_handler = [&](http::response res) -> void {
    if (res.reqwest().get_header("accept-encoding").find("gzip") ==
        std::string::npos) {
      res.set_content("Error: gzip is not supported by this browser",
                      "text/plain");
    } else {
      res.set_header("Content-Encoding", "gzip");
      // COEP and COOP headers, required by pyodide (python interpreter)
      res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
      res.set_header("Cross-Origin-Opener-Policy", "same-origin");
      res.set_content(reinterpret_cast<const char *>(index_html_gz),
                      index_html_gz_len, "text/html; charset=utf-8");
      // res.set_content(reinterpret_cast<std::>(index_html_gz),
      // index_html_gz_len, "text/html; charset=utf-8");
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
  route_.post("/completions", [](http::response res) {
    std::thread{completions_chat_handler, std::move(res)}.detach();
  });
  route_.post("/v1/completions", [](http::response res) {
    std::thread{completions_chat_handler, std::move(res)}.detach();
  });
  // chat
  route_.post("/chat/completions", [](http::response res) {
    std::thread{completions_chat_handler, std::move(res)}.detach();
  });
  route_.post("/v1/chat/completions", [](http::response res) {
    std::thread{completions_chat_handler, std::move(res)}.detach();
  });

  AVLLM_LOG_INFO("Server is started at http://127.0.0.1:%d\n", srv_port);
  http::start_server(srv_port, route_);

  LOG("\n");

  llama_model_free(model);
  return 0;
}
