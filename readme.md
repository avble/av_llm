# Overview

- Gen AI stuff

# Compilation 

 | OS      | Download link |
 |---------|---------------|
 | macOS   | T.B.U         |
 | Windows | T.B.U         |
 | Linux   | T.B.U         |


## GPU based package 
To take avantage of GPU's memory and computation.
Support various GPU's library/platform.

 | OS      | CUDA | VULKAN | SYCL |
 |---------|------|--------|------|
 | Windows |      |        |      |
 | Linux   |      |        |      |
 | macOS   | x    | x      | x    |


# Compilation

'''
$ # clone source code
$ cmake -B build && cmake --build build
'''

# Quick started

```shell
$av_llm chat <path to module .gguf file>
```

```shell
$av_llm serve <path to module .gguf file>
```

![demo-1](https://github.com/avble/av_llm/blob/main/image/demo_4.png?raw=true)


## Webassembly

[In-browser](https://avble.github.io/wav_llm/) LLM inference 

# Main components

- Web UI: Provide a simple web UI interface to explore/experiment (borrowed from @llama.cpp project)
- CLI: An lightweight and simple command-line-interface
- A lightweight OpenAI API compatible server: [av_connect http server](https://github.com/avble/av_connect.git) in C++
- LLM Inference engine: by [llama.cp](https://github.com/ggerganov/llama.cpp.git)


# Some demo
+ infill
+ rerank
+ server completion 

# Models

## Source Code
### Fill-in-middle (fim) model

Qwen2.5-Coder-1.5B-Q8_0-GGUF

### Code instruct

## General chat

## List models

| Model                                                                                             | ~GB   | Tags          | Remark  |
|---------------------------------------------------------------------------------------------------|-------|---------------|---------|
| (Qwen3-4B)[https://huggingface.co/Qwen/Qwen3-4B-GGUF]                                             | ~3 Gb | coding, think | Q4, K_M |
| (Qwen2.5-Coder-3B-Q8_0-GGUF)[https://huggingface.co/ggml-org/Qwen2.5-Coder-3B-Instruct-Q8_0-GGUF] |       | coding, FIM   |         |
|                                                                                                   |       |               |         |

# Supported model

This application is built on the top of [llama.cpp](https://github.com/ggerganov/llama.cpp), so it should work any model which the [llama.cpp](https://github.com/ggerganov/llama.cpp) supports

- LLaMA 1
- LLaMA 2
- LLaMA 3
- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

# Known limitation

- The performance of inference in web is poor

# Future work

- Support more AI tasks
  

# Note

This is demonstration version, some issues or error checking is not fully validated.
<br>
Contact me via `avble.harry dot gmail.com` if any
