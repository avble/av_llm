---
title: Overview
description: Avble Overview.
slug: /docs/overview
---

## Installation

- Or download from github

## Quick started

### start an interactive chat

```
$ av_llm ggml-org:Qwen2.5-Coder-3B-Q8_0-GGUF
```

OR

```
$ av_llm <path-to-your-gguf-file>  // point to .gguf file
```

OR

```
$ av_llm <url-path-to-your-gguf-file> // url for dowloaind gguf file
```

### start a server

```
$ av_llm server ggml-org:Qwen2.5-Coder-3B-Q8_0-GGUF
```

OR

```
$ av_llm server <path-to-your-gguf-file>  // point to .gguf file
```

#### Open WebUI

```
http://127.0.0.1:8080
```

## model-path

- can be a path of .gguf file `./models/model-1.gguf`
- can be the url to download .gguf file in huggingface
- Or can be use the model name of following table, which is internally map to a url

| Model Name                          | Quantization    |
| ----------------------------------- | --------------- |
| ggml-org:Qwen2.5-Coder-3B-Q8_0-GGUF | Q8              |
| Qwen:Qwen2.5-Coder-3B-Instruct-GGUF | default: Q4_K_M |
| Qwen:Qwen3-4B-GGUF                  | default: Q4_K_M |
