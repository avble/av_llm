# Motivation
* The Large language models (such as GPT, LLAMA, etc) have evolutionized the NPL
* [llama.cpp](https://github.com/ggerganov/llama.cpp.git) has created by ggerganov in Pure C/C++. I might be applied on various platform (embedded device, cloud, ...)
* This tool is created as an essential stuff make it simple to explore/research the capability of applying LLM model in various domain and various application.

# Note
This is demonstration version, some issues or error checking is not fully validated.
<br>
Contact me via `avble.harry dot gmail.com` if any

# Features
* OpenAI API compatible server (chat and completion endpoint)
* Simple Web UI for explore/debug
* Currently, it only supports chat application

## Some snapshot
![demo-1](https://github.com/avble/av_llm/blob/main/image/demo_1.JPG?raw=true)
![demo-2](https://github.com/avble/av_llm/blob/main/image/demo_2.JPG?raw=true)


# Quick started
Obtain the latest container from [docker hub](https://hub.docker.com/)
``` shell
docker image pull harryavble/av_llm
```

Run from docker
``` shell
docker run -p 8080:8080  harryavble/av_llm:latest
```

Access to Web interface at http://127.0.0.1:8080

# Tech-stack
* OpenAI API compatible server: [av_connect http server](https://github.com/avble/av_connect.git)
* LLM Inference: [llama.cp](https://github.com/ggerganov/llama.cpp.git)
* Web UI: Provide a simple web UI interface to explore/experiment

# Compile and run
T.B.D

# Supported model
This application is built on the top of [llama.cpp](https://github.com/ggerganov/llama.cpp), so it should work any model which the [llama.cpp](https://github.com/ggerganov/llama.cpp) supports 
* LLaMA 1
* LLaMA 2
* LLaMA 3
* [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
* [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
* [DBRX](https://huggingface.co/databricks/dbrx-instruct)
* [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
* [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

## Download model and run
``` cmd
docker run -p 8080:8080 -v $your_host_model_folder:/work/model av_llm ./av_llm -m /work/model/$your_model_file
```


# Future work
* Support more LLM models
* Support more OpenAI API server
* Support more application 

# Reference
* https://platform.openai.com/docs/api-reference/introduction
