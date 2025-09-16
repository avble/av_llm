#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

extern "C" {
void av_llm_init(const char* model_path);
void av_llm_set_prompt(std::vector<int32_t> prompt_tokens);
int av_llm_get_next_token();
void av_llm_debug(std::vector<int32_t> debug_tokens);
}

namespace py = pybind11;

PYBIND11_MODULE(avllm, m) {
  m.doc() = "Python bindings for AV LLM C++ backend";

  m.def(
      "init",
      [](const std::string& config_path) { av_llm_init(config_path.c_str()); },
      "Initialize the LLM with config path");

  m.def(
      "set_prompt",
      [](const std::vector<int32_t>& prompt_tokens) {
        try {
          av_llm_set_prompt(prompt_tokens);
        } catch (const std::exception& e) {
          std::cerr << "Exception in set_prompt: " << e.what() << std::endl
                    << std::flush;
        }
      },
      "Set the prompt tokens for LLM");

  m.def("get_next_token", &av_llm_get_next_token,
        "Get the next token from LLM");

  m.def(
      "debug",
      [](const std::vector<int32_t>& tokens) {
        try {
          av_llm_debug(tokens);
        } catch (const std::exception& e) {
          std::cerr << "Exception in debug: " << e.what() << std::endl
                    << std::flush;
        }
      },
      "Debug function");
}
