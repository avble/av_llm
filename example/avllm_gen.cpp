/*
given a promt from command-line or reading from file
generate the tokens
*/

#include <inttypes.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#ifdef _MSC_VER
#include <ciso646>
#endif

static void print_usage(int, char** argv) {
  printf("\nexample usage:\n");
  printf(
      "%15s -m model.gguf -input prompt\n"
      "                              -input @file\n",
      argv[0]);
  printf("\n");
}

// global option
std::string model_path;
std::string prompt;
int top_k = 20;
float temperature = 1.0;
float top_p = 1.8;
float min_p = 0.05;

int main(int argc, char** argv) {
  common_params params;

  [&argc, &argv]() {  // parsing the argument
    int i = 0;
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-m") == 0) {
        if (i + 1 < argc) {
          model_path = argv[++i];
        } else {
          print_usage(1, argv);
        }
      } else if (strcmp(argv[i], "-input") == 0) {
        if (i + 1 < argc) {
          prompt = argv[++i];
        } else {
          print_usage(1, argv);
        }
      } else if (strcmp(argv[i], "-h") == 0) {
        print_usage(1, argv);
        exit(0);
      }
    }
  }();
  if (model_path == "" or prompt == "") {
    print_usage(1, argv);
    return 1;
  }

  if (prompt.size() > 1 and prompt[0] == '@') {
    std::string input_file_path = prompt.substr(1);
    std::ifstream in_f(input_file_path);
    if (not in_f.is_open()) {
      std::cerr << "file: " << input_file_path << " not found " << std::endl;
      exit(-1);
    }
    std::ostringstream osstream;
    osstream << in_f.rdbuf();
    prompt = osstream.str();
  }

  ggml_backend_load_all();

  // model initialized
  llama_model* model = []() -> llama_model* {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    return llama_model_load_from_file(model_path.c_str(), model_params);
  }();
  if (model == nullptr) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return 1;
  }

  // context initialize
  llama_context* ctx = [&model]() -> llama_context* {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                         NULL, 0, true, true);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.no_perf = false;
    ctx_params.n_ctx = 10000;
    ctx_params.n_batch = 5000;

    return llama_init_from_model(model, ctx_params);
  }();
  if (ctx == nullptr) {
    fprintf(stderr, "%s: error: failed to create the llama_context\n",
            __func__);
    return -1;
  }

  // initialize the sampler
  llama_sampler* smpl = []() {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, min_p));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    // llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));
    return smpl;
  }();
  if (smpl == nullptr) {
    fprintf(stderr, "%s: error: could not create sampling\n", __func__);
    return 1;
  }

  // decode the prompt
  {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (vocab == nullptr) {
      fprintf(stderr, "%s: failed to get vocal from model \n", __func__);
      return -1;
    }

    const int n_prompt_tokens = -llama_tokenize(
        vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);

    if (llama_tokenize(vocab, prompt.data(), prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), true,
                       true) < 0) {
      fprintf(stderr, "%s: failed to tokenize the prompt \n", __func__);
      return -1;
    }

    if (false) {  // print token
      std::cout << "tokens: \n";
      for (auto token : prompt_tokens) {
        char buf[120];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
          fprintf(stderr, "%s: error: failed to tokenize \n", __func__);
          return 1;
        }
        std::string s(buf, n);
        std::cout << s;
      }
      std::cout << "\n";
    }

    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch)) {
      fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
      return -1;
    }
  }

  std::cout << "Generate text:\n";
  {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const auto t_main_start = ggml_time_us();
    llama_token new_token_id;
    int num_token = 0;
    while (num_token++ < 2000)  // generate 1000 tokens
    {
      new_token_id = llama_sampler_sample(smpl, ctx, -1);

      // is it an end of generation?
      if (llama_vocab_is_eog(vocab, new_token_id)) break;

      char buf[128];
      int n =
          llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
      if (n < 0) {
        fprintf(stderr, "%s: error: failed to convert token to piece\n",
                __func__);
        break;
      }
      std::string text(buf, n);
      printf("%s", text.c_str());
      fflush(stdout);

      llama_batch batch = llama_batch_get_one(&new_token_id, 1);
      // evaluate the current batch with the transformer model
      if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        return -1;
      }
    }
  }

  if (true) {
    printf("\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
  }

  llama_sampler_free(smpl);
  llama_free(ctx);
  llama_model_free(model);
  return 0;
}
