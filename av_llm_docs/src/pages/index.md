---
title: Overview
description: Avble Overview.
slug: /docs/overview
---

## Installation

- TBD (will download from github)

## Quick started

### start an interactive chat

```
$ av_llm qween3-1.7b
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
$ av_llm serve qween3-1.7b
```

OR

```
$ av_llm serve <path-to-your-gguf-file>  // point to .gguf file
```

#### Open WebUI

```
http://127.0.0.1:8080
```

## model-path

- can be a path of .gguf file `./models/model-1.gguf`
- can be the url to download .gguf file in huggingface
- Or can be use the model name of following table, which is internally map to a url

| Model Name          | Remarks |
| ------------------- | ------- |
| phi-3-mini-4k       |         |
| qween3-1.7b         |         |
| qween2.5-coder-0.5b |         |
| qween2.5-coder-3b   |         |
