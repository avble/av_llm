## Overview

- Develop a C++ server for serving the LLM inference
- LLM inference engine based on [llama.cpp]
- Perdiodically sync with upstream [llama.cpp]
- Explore the LLM model

## Demos
* Demo chat with Jan AI agent
<img src="docs/images/01_chat_jan_ai.gif" alt="Demo chat" width="400" />
<details>
<summary>
Demo function calling
 </summary>
<img src="docs/images/01_function_call.gif" alt="Demo function call" width="400" />
 </details>

 <details>
 <summary>
Demo Code Completion
</summary>  
<img src="docs/images/02_code_completion.gif" alt="Demo code completion" width="400" />
</details>

## What it has?
### [Endpoint] C++ server for serving LLM inference
<details>
 <summary>
[open-ai] Completions
  </summary>
 
   - [x] Default
   - [x] stream
</details>
<details>
 <summary>
[open-ai] Chat completions
  </summary>
 
   - [x] Default
   - [x] stream
   - [ ] Image input
   - [x] function
   - [ ] Logbrobs
</details>
<details>
 <summary>
[open-ai] Models 
  </summary>
 
   - [x] list models  
   - [x] retrieve model 
   - [ ] delete a model
</details>

[open-ai] Embeddings 

<details>
 <summary>
ollama 
  </summary>
 
   - [x] /api/tags
   - [x] /api/show  
   - [x] /api/chat 
   - [ ] /api/generate
</details>

<details>
 <summary> FIM (Fill-In-Middle) </summary>
 
- [x] File-level 
- [ ] Rep-level
 </details>
 

## Models
[details](docs/model.md)

| Model                                                                                             | ~GB   | Tags          | Remark  |
|---------------------------------------------------------------------------------------------------|-------|---------------|---------|
| Qwen3-8B                                                                                          |       | think, MCP    |         |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF)                                             | ~3 Gb |               | Q4, K_M |
| [Qwen2.5-Coder-3B-Q8_0-GGUF](https://huggingface.co/ggml-org/Qwen2.5-Coder-3B-Instruct-Q8_0-GGUF) |       | chat, coding, FIM   |         |
|                                                                                                   |       |               |         |

## work log 
| Date       | Work log                                                                 |
|------------|--------------------------------------------------------------------------|
| 2025-07-24 | Sync with upstream of [llama.cpp]                    |
| 2025-07-23 | Finalize initial base version: server, command line    |


### Some tools 
|Item                 |Brief                                              |link                           |
|-------------------- |-------------------------------------------------- |------------------------------ |
|gen                  |given input, generate the sequence of text         |[example/avllm_gen.cpp](example/avllm_gen.cpp) |
|chat                 |given chatML format, generate the sequence of text |[example/avllm_gen.cpp](example/avllm_chat.cpp) |
|embedding            |given input, generate a embedding vector           |[example/avllm_embedding.cpp](example/avllm_embedding.cpp) |


## Installation
<details>
<summary> Pre-built </summary>
# Pre-built 

 | OS      | Download link |
 |---------|---------------|
 | macOS   | T.B.U         |
 | Windows | T.B.U         |
 | Linux   | T.B.U         |


### GPU based package 
To take avantage of GPU's memory and computation.
Support various GPU's library/platform.

 | OS      | CUDA | VULKAN | SYCL |
 |---------|------|--------|------|
 | Windows |      |        |      |
 | Linux   |      |        |      |
 | macOS   | x    | x      | x    |

</details>

## Compilation

``` shell
T.B.U
```

## Main components
- Web UI: Provide a simple web UI interface to explore/experiment (borrowed from @llama.cpp project)
- CLI: An lightweight and simple command-line-interface
- A lightweight OpenAI API compatible server: [av_connect http server](https://github.com/avble/av_connect.git) in C++
- LLM Inference engine: by [llama.cp](https://github.com/ggerganov/llama.cpp.git)

## Future work
- Support more AI tasks
  
## Note
This is demonstration version, some issues or error checking is not fully validated.
<br>
Contact me via `avble.harry dot gmail.com` if any
