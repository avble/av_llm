## Overview

- Gen AI stuff

## News

- Demo MCP: [document](./docs/model.md), search for MCP
- Demo FIM: [document](./docs/model.md), search for FIM
- Demo Code Completion: [document](./docs/model.md), search for code completion

## What it has?
### Lightweight OpenAI compatible server
<details>
 <summary>
[Endpoint] Create Chat completion
  </summary>
 
   - [x] Default
   - [x] stream
   - [ ] Image input
   - [ ] function
   - [ ] Logbrobs
</details>
<details>
 <summary>
[Endpoint] Models 
  </summary>
 
   - [x] list models  
   - [x] retrieve model 
   - [ ] delete a model
</details>

<details>
 <summary> [Endpoint] FIM (Fill-In-Middle) </summary>
 
- [x] File-level 
- [ ] Rep-level
 </details>
 
### Some examples
|Item                 |Brief                                              |link                           |
|-------------------- |-------------------------------------------------- |------------------------------ |
|gen                  |given input, generate the sequence of text         |[example/avllm_gen.cpp](example/avllm_gen.cpp) |
|chat                 |given chatML format, generate the sequence of text |[example/avllm_gen.cpp](example/avllm_chat.cpp) |
|embedding            |given input, generate a embedding vector           |[example/avllm_embedding.cpp](example/avllm_embedding.cpp) |


## Models
[details](docs/model.md)

| Model                                                                                             | ~GB   | Tags          | Remark  |
|---------------------------------------------------------------------------------------------------|-------|---------------|---------|
| Qwen3-8B                                                                                          |       | think, MCP    |         |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF)                                             | ~3 Gb |               | Q4, K_M |
| [Qwen2.5-Coder-3B-Q8_0-GGUF](https://huggingface.co/ggml-org/Qwen2.5-Coder-3B-Instruct-Q8_0-GGUF) |       | chat, coding, FIM   |         |
|                                                                                                   |       |               |         |


## Quick started

```shell
$av_llm chat <path to module .gguf file>
```

```shell
$av_llm serve <path to module .gguf file>
```

![demo-1](image/demo_4.png?raw=true)

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
