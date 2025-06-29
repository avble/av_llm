#ifndef _AVLLM_MODEL_H_
#define _AVLLM_MODDEL_H_

#include <string>
#include <unordered_map>

static std::unordered_map<std::string, std::string> pre_config_model;

static void pre_config_model_init()
{
    // Qwen model
    // pre_config_model["qween2.5-coder-3b"] =
    //     "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf";

    // pre_config_model["qween2.5-coder-0.5b"] =
    //     "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";

    pre_config_model["qween3-1.7b"] = "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf";

    // tinyllama (error)
    // pre_config_model["tinyllama-1.1b"] =
    //     "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.2-GGUF/resolve/main/ggml-model-q4_0.gguf";

    // gemma ()

    // llama

    // phi3
    pre_config_model["phi-3-mini-4k"] =
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf";
}

#endif