# Qwen3-8B
## Tool
```sh
$ avllm_gen.exe -m Qwen3-8B -input @file_1 
```

<details>
<summary>
what is the weather like?
</summary>

- given text

``` 
<|im_start|>system
You are a helpful assistant that can use tools to get information for the user.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "get_weather", "description": "Get current weather information for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to use"}}, "required": ["location"]}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's the weather like in New York?<|im_end|><|im_start|>

```

- output
```
<tool_call>
{"name": "get_weather", "arguments": {"location": "New York", "unit": "fahrenheit"}}
</tool_call>
```

</details>

# Qwen2.5-Coder-Instructor
## Chat
prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

## FIM
{
  "<|fim_prefix|>": 151659, 
  "<|fim_middle|>": 151660, 
  "<|fim_suffix|>": 151661, 
  "<|fim_pad|>": 151662, 
  "<|repo_name|>": 151663, 
  "<|file_sep|>": 151664, 
  "<|im_start|>": 151644, 
  "<|im_end|>": 151645
}

## Tool

```
prompt = '<|fim_prefix|>' + prefix_code + '<|fim_suffix|>' + suffix_code + '<|fim_middle|>'
```

``` python
from transformers import AutoTokenizer, AutoModelForCausalLM
# load model
device = "cuda" # the device to load the model onto

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B")
MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-32B", device_map="auto").eval()

input_text = """<|fim_prefix|>def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    <|fim_suffix|>
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)<|fim_middle|>"""

model_inputs = TOKENIZER([input_text], return_tensors="pt").to(device)

# Use `max_new_tokens` to control the maximum output length.
generated_ids = MODEL.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=False)[0]
# The generated_ids include prompt_ids, we only need to decode the tokens after prompt_ids.
output_text = TOKENIZER.decode(generated_ids[len(model_inputs.input_ids[0]):], skip_special_tokens=True)

print(f"Prompt: {input_text}\n\nGenerated text: {output_text}")
```

# Godegema

# Codestral

# Deepseek coder

# Qwen3-embedding
+ sequence(sentence)-level decoding
+ 

# (Qwen3-4B-GGUF)[https://huggingface.co/Qwen/Qwen3-4B-GGUF]

``` i.e chat
> Who are you /no_think

<think>

</think>

I am Qwen, a large-scale language model developed by Alibaba Cloud. [...]

> How many 'r's are in 'strawberries'? /think

<think>
Okay, let's see. The user is asking how many times the letter 'r' appears in the word "strawberries". [...]
</think>

The word strawberries contains 3 instances of the letter r. [...]

```

## Reference
- https://qwenlm.github.io/blog/qwen3-embedding/
- https://arxiv.org/abs/2505.09388
- https://arxiv.org/pdf/2409.12186
- https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e
- https://huggingface.co/collections/google/codegemma-release-66152ac7b683e2667abdee11
- https://qwen.readthedocs.io/en/latest/
