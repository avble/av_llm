# Overview

- Gen AI stuff

# Installation

| OS      | Download link |
| ------- | ------------- |
| macOS   | T.B.U         |
| Windows | T.B.U         |
| Linux   | T.B.U         |

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

# Models

| Model                      | ~GB     | Tags         | Linked                                                              | Remark |
| -------------------------- | ------- | ------------ | ------------------------------------------------------------------- | ------ |
| Devstral-Small-2505 (gguf) | 13 GB   | coding       | https://huggingface.co/mistralai/Devstral-Small-2505_gguf/tree/main |        |
| Gema 2B                    | 10Gb    |              | https://huggingface.co/google/gemma-1.1-2b-it-GGUF/tree/main        |        |
| Gemma-3 4b                 | 3.16GB  | general chat | https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/tree/main | v      |
| Phi-4                      | > 3.5GB |              | https://huggingface.co/microsoft/phi-4-gguf/tree/main               | x      |
| Phi-3 Inst                 | 2.2 GB  |              | https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf        | x      |
| Binet                      | 1.19G   | general chat | https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/tree/main  | v      |
| Qwen3-4B                   | ~3 Gb   | coding       | https://huggingface.co/Qwen/Qwen3-4B-GGUF                           | v      |
|                            |         |              |                                                                     |        |

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
