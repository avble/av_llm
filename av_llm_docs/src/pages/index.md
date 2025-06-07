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

## models

- Or can be use the model name of following table, which is internally map to a url

| Model Name          | Remarks                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| phi-3-mini-4k       | - chat <br/>`av_llm chat phi-3-mini-4k` <br/> - server <br/>`av_llm serve phi-3-mini-4k`             |
| qween3-1.7b         | - chat <br/>`av_llm chat qween3-1.7b` <br/> - server <br/>`av_llm serve qween3-1.7b`                 |
| qween2.5-coder-0.5b | - chat <br/>`av_llm chat qween2.5-coder-0.5b` <br/> - server <br/>`av_llm serve qween2.5-coder-0.5b` |
| qween2.5-coder-3b   | - chat <br/>`av_llm chat qween2.5-coder-3b` <br/> - server <br/>`av_llm serve qween2.5-coder-3b`     |

- can be a path of .gguf file `./models/model-1.gguf`
- can be the url to download .gguf file in huggingface
