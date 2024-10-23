# Motivation
* LLM model (such as GPT, LLAMA, etc) has been provided
* [llama.cpp](https://github.com/ggerganov/llama.cpp.git) has created by ggerganov in Pure C/C++. I might be applied on various platform (embedded device, cloud, ...)
* It is essential to have a simple tool to explore/research the capability of applying LLM model in various domain.

# Note
This is demonstration version, some issues or error checking is not fully validated.
Contact me via avble.harry dot gmail.com if any

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

# Future work
* Support more LLM model
* Support more application 

# Reference
* https://platform.openai.com/docs/api-reference/introduction
