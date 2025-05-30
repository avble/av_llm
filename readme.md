# Overview

- Gen AI stuff

# Tech-stack

- A lightweight OpenAI API compatible server: [av_connect http server](https://github.com/avble/av_connect.git) in C++
- LLM Inference engine: by [llama.cp](https://github.com/ggerganov/llama.cpp.git)
- Web UI: Provide a simple web UI interface to explore/experiment (borrowed from @llama.cpp project)

## A snapshot

![demo-1](https://github.com/avble/av_llm/blob/main/image/demo_4.png?raw=true)

## Webassembly 
![In-browser](https://avble.github.io/wav_llm/) LLM inference

# Compile and run

```
$ cmake -B build && cmake --build build
$ build/av_llm -m <path to gguf file>
```

Access to Web interface at http://127.0.0.1:8080

# Chat

# Supported model

- LLaMA 1
- LLaMA 2
- LLaMA 3
- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
  This application is built on the top of [llama.cpp](https://github.com/ggerganov/llama.cpp), so it should work any model which the [llama.cpp](https://github.com/ggerganov/llama.cpp) supports

## Download model and run

```cmd
docker run -p 8080:8080 -v $your_host_model_folder:/work/model av_llm ./av_llm -m /work/model/$your_model_file

```

# UI

Should work with below UI

- [huggingface/chat-ui](https://github.com/huggingface/chat-ui)
-

# Future work

- Support more AI tasks

# Note

This is demonstration version, some issues or error checking is not fully validated.
<br>
Contact me via `avble.harry dot gmail.com` if any

# Reference

- https://platform.openai.com/docs/api-reference/introduction


# Edge device framework
## pytorch edge
https://pytorch.org/executorch/stable/intro-how-it-works
https://github.com/pytorch/executorch
## tensorflow lite


# Others
https://dreampuf.github.io/GraphvizOnline
https://github.com/lutzroeder/netron
