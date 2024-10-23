# Model
* llama
* Mistral 7B
* falcon
* dbrx-instruct
* Chinese-LLaMA-Alpaca
* vigogne 


# openai API
## Text generation
API `/v1/chat/completions`

Generate prose
``` shell
curl "127.0.0.1:12345/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming."
            }
        ]
    }'
```
``` json - llama/cpp
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": " In loops, code breathes deep,\n\nA function calls, then lets go,\n\nRecursion's dance.",
        "role": "assistant"
      }
    }
  ],
  "created": 1729146865,
  "model": "gpt-4o",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 28,
    "prompt_tokens": 21,
    "total_tokens": 49
  },
  "id": "chatcmpl-MF58NtgHpiE8WYH3krU1SW8O7o6tTuoi"
}
```


### stream
``` shell
curl "127.0.0.1:8080/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming."
            }
        ],
        "stream": true
    }'
```

``` json
{
  "choices": [
    {
      "finish_reason": null,
      "index": 0,
      "delta": {
        "content": "."
      }
    }
  ],
  "created": 1729147198,
  "id": "chatcmpl-HJocwP9MnoHaBFJuCTn0uxnErxCvgqZv",
  "model": "gpt-4o",
  "object": "chat.completion.chunk"
}
````

last chunk
``` json (llama-cpp)
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "delta": {}
    }
  ],
  "created": 1729147198,
  "id": "chatcmpl-HJocwP9MnoHaBFJuCTn0uxnErxCvgqZv",
  "model": "gpt-4o",
  "object": "chat.completion.chunk",
  "usage": {
    "completion_tokens": 27,
    "prompt_tokens": 21,
    "total_tokens": 48
  }
}

```



Analyze images
``` shell
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
        ]
      }
      }
  }'
```

Generate JSON
``` shell
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o-2024-08-06",
    "messages": [
      {
        "role": "system",
        "content": "You extract email addresses into JSON data."
      },
      {
        "role": "user",
        "content": "Feeling stuck? Send a message to help@mycompany.com."
      }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "email_schema",
        "schema": {
            "type": "object",
            "properties": {
                "email": {
                    "description": "The email address that appears in the input",
                    "type": "string"
                }
            },
            "additionalProperties": false
        }
      }
    }
  }'
```

## Image generation
API `v1/images/generations`

Generate an image
``` shell
curl https://api.openai.com/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "dall-e-3",
    "prompt": "a white siamese cat",
    "n": 1,
    "size": "1024x1024"
  }'
```

Edit an image
``` shell
curl https://api.openai.com/v1/images/edits \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F model="dall-e-2" \
  -F image="@sunlit_lounge.png" \
  -F mask="@mask.png" \
  -F prompt="A sunlit indoor lounge area with a pool containing a flamingo" \
  -F n=1 \
  -F size="1024x1024"
```

## vision - Vector embeddings
API `/v1/embeddings`

* Search 
* Clustering 
* Recommendations 
* Anomaly detection
* Diversity measurement
* Classification 


## Text to speech
API `v1/audio/speech`

``` shell
curl https://api.openai.com/v1/audio/speech \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Today is a wonderful day to build something people love!",
    "voice": "alloy"
  }' \
  --output speech.mp3
```
Note: 
It is chunk transfer


## Speech to text
### transcription
API `v1/audio/transcriptions`

``` shell
curl --request POST \
  --url https://api.openai.com/v1/audio/transcriptions \
  --header "Authorization: Bearer $OPENAI_API_KEY" \
  --header 'Content-Type: multipart/form-data' \
  --form file=@/path/to/file/audio.mp3 \
  --form model=whisper-1
```
### translation
API `v1/audio/translations`
``` shell
curl --request POST \
  --url https://api.openai.com/v1/audio/translations \
  --header "Authorization: Bearer $OPENAI_API_KEY" \
  --header 'Content-Type: multipart/form-data' \
  --form file=@/path/to/file/german.mp3 \
  --form model=whisper-1

```

## moderation
API `v1/moderations`

``` shell
curl https://api.openai.com/v1/moderations \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "omni-moderation-latest",
    "input": "...text to classify goes here..."
  }'

```

# OpenAI UI
* https://github.com/imoneoi/openchat-ui


# framework serving OpenAI 
* https://github.com/vllm-project/vllm/tree/main
* https://github.com/sgl-project/sglang
* https://github.com/InternLM/lmdeploy
* https://github.com/NVIDIA/TensorRT-LLM
* 


## Reference
* https://platform.openai.com/docs/guides/text-generation

# LLM inference
| Name | language | openai server? |
|------| -------- | ------------ |
| transformers | Python | | v
| Text Generation Inference | Python, Rust | |
| gpt-fast | python| |
| TensorRT-LLM | C++ | | v
| vllm | Python| | 
| llama.cpp | C++ | | v
| ggml | C, C++| | it includes llama.cpp and wisher.cpp and tranformer
| ctransformers | python biding of ggml | |
| DeepSpeed | ??? | |
| FastChat | ??? |  |
| lightllm | python| | just interface with other model|
| lmdeploy | C++, python | |  
| PowerInfer | C++ | | The performance is quite impressive | v

## OpenAI interface
* transformer
No, built-in. But libraries: 
** https://github.com/jquesnelle/transformers-openai-api/tree/master (python)
** https://github.com/mesolitica/transformers-openai-api (python-fastapi)
** 

* TensorRT-LLM
Supported library
https://github.com/npuichigo/openai_trtllm

* vllm
Yes, built-in library.
Python, and on the top of fastapi

# openai API client
Support python and node (javascript)

# NLP
Hugging Face Transformers
spaCy
NLTK


# Chat template

# Chat template

* openAI API
``` json
[
  {
    "role": "user",
    "content": "what is your name"
  },
  {
    "role": "assistant",
    "content": " As an AI, I dont have a personal name, but you can refer to me as your AI Assistant. Im designed to help answer your questions and provide information."
  },
  {
    "role": "user",
    "content": "good morning"
  }
]
```

## transforms LLM

* mistra
``` shell
text = "<s>[INST] What is your favourite condiment? [/INST]"
"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
"[INST] Do you have mayonnaise recipes? [/INST]"
```

* Zephyr
input
``` shell
<|user|>
Hello, how are you?</s>
<|assistant|>
I'm doing great. How can I help you today?</s>
<|user|>
I'd like to show off how chat templating works!</s>
```

output
``` shell
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

## reference
* https://huggingface.co/docs/transformers/main/en/chat_templating
* 

# Prompt engineer
[1] https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/prompt-engineering
[2] https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-prompt-engineering.html




# llama cpp API
## tokenize prompt
``` cpp
    tokens_list = ::llama_tokenize(ctx, params.prompt, true);
```

## print token by token
``` cpp
    for (auto id : tokens_list) {
        LOG("%s", llama_token_to_piece(ctx, id).c_str());
    }
```

# openAI API
``` json
curl 127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'


``` json
[
  {
    "role": "user",
    "content": "what is your name"
  },
  {
    "role": "assistant",
    "content": " As an AI, I dont have a personal name, but you can refer to me as your AI Assistant. Im designed to help answer your questions and provide information."
  },
  {
    "role": "user",
    "content": "good morning"
  }
]

```

curl 127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'

< HTTP/1.1 200 OK
< Access-Control-Allow-Origin: 
< Content-Type: text/event-stream
< Keep-Alive: timeout=5, max=5
< Server: llama.cpp
< Transfer-Encoding: chunked
< 
``` json
{
  "choices": [
    {
      "finish_reason": null,
      "index": 0,
      "delta": {
        "content": "?"
      }
    }
  ],
  "created": 1728956079,
  "id": "chatcmpl-PtbptoUnzExyHvnL1PX7sTJ0xNS12yBN",
  "model": "gpt-4o",
  "object": "chat.completion.chunk"
}
```

< HTTP/1.1 200 OK
< content-type: text/event-stream
< transfer-encoding: chunked
< server: av_connect
``` json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1728958151,
  "model": "gpt-4o-mini",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "user",
        "content": " humans"
      },
      "logprobs": null,
      "finish_reason": null
    }
  ]
}
```



### automated pipeline

``` python
from transformers import pipeline

pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # Print the assistant's response
```

``` json
{
  "role": "assistant",
  "content": "Matey, Im afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o grog, a savory bowl o stew, or a delicious loaf o bread. But helicopters, they be for transportin and movin around, not for eatin. So, Id say none, me hearties. None at all."
}
```

### how llama cpp deal with various chat template
<OpenAI message>   ==> [llama cpp inference] ==> [chat template (which one??)] (openAI )

# LLM inference
autoregressive, GPT (generate pretrain training), 

* context
* sampling
* decode


* LLM library
PyTorch
TensorFlow

# Other
* langchan
* generate python biding via C/C++ header file

# Reference
[1] https://platform.openai.com/docs/api-reference/assistants-streaming-v1
[2] https://github.com/godaai/llm-inference
[3] https://html.spec.whatwg.org/multipage/server-sent-events.html#server-sent-events

# Reference LLM
[1] https://jalammar.github.io/illustrated-transformer/
[2] https://www.omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp/
[3] https://www.omrimallis.com/posts/techniques-for-kv-cache-optimization/
[4] https://github.com/DefTruth/Awesome-LLM-Inference
[5] https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
[6] https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca ( embedded word)
[7] https://www.kaggle.com/competitions/word2vec-nlp-tutorial
[8] https://jaroncollis.medium.com/
[9] https://jalammar.github.io/illustrated-transformer/
