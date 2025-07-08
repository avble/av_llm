# Qwen3-8B
## Think

## Function calling 
```sh
$ avllm_gen.exe -m Qwen3-8B -input @file_1 
```

<details>
<summary>
what is the weather like?
</summary>

<details>
<summary>
    given text
</summary>
    
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
</details>

<details>
<summary>
gen function call
</summary>
    
```
<tool_call>
{"name": "get_weather", "arguments": {"location": "New York", "unit": "fahrenheit"}}
</tool_call>
```
</details>
</summary>

Other tools
```
# Home automation 
{"name":"turn_off","description":"turn off a device","parameters":{"type":"object","properties":{"name":{"type":"string","description":"turn off a device, e.g. light 01, light 02, living room"}},"required":["name"]}}
{"name":"turn_on","description":"turn on a device","parameters":{"type":"object","properties":{"name":{"type":"string","description":"turn on a device, e.g. light 01, light 02, living room"}},"required":["name"]}}
{"name":"get_all_devices","description":"Get all devices","parameters":{}}
{"name":"get_devices_by_name","description":"get device by names","parameters":{"type":"object","properties":{"name":{"type":"string","description":"get device by name, e.g. light 01, light 02, lights"}},"required":["name"]}}
```

</details>

# Qwen2.5-Coder-Instructor
## Chat

- given chatML format messages
``` json
[
  {
    "role": "system",
    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "write a quick sort algorithm in python"
  }
]
```
- expected output

Certainly! Quick sort is a popular and efficient sorting algorithm that uses a divide-and-conquer approach. Below is a simple implementation of the quick sort
 algorithm in Python:

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]  # Choose the middle element as the pivot
        left = [x for x in arr if x < pivot]  # Elements less than the pivot
        middle = [x for x in arr if x == pivot]  # Elements equal to the pivot
        right = [x for x in arr if x > pivot]  # Elements greater than the pivot
        return quick_sort(left) + middle + quick_sort(right)

# Example usage:
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = quick_sort(arr)
print("Sorted array:", sorted_arr)

```

<details>
<summary>
more..
</summary>
### Explanation:
1. **Base Case**: If the array has 0 or 1 element, it is already sorted, so we return it as is.
2. **Pivot Selection**: We choose the middle element of the array as the pivot. This is a simple choice, but other strategies like choosing the first, last, o
r a random element can also be used.
3. **Partitioning**: We create three lists:
   - `left`: Contains elements less than the pivot.
   - `middle`: Contains elements equal to the pivot.
   - `right`: Contains elements greater than the pivot.
4. **Recursive Sorting**: We recursively apply the quick sort algorithm to the `left` and `right` lists and concatenate the results with the `middle` list.
</details>

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


## Reference
- https://qwenlm.github.io/blog/qwen3-embedding/
- https://arxiv.org/abs/2505.09388
- https://arxiv.org/pdf/2409.12186
- https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e
- https://huggingface.co/collections/google/codegemma-release-66152ac7b683e2667abdee11
- https://qwen.readthedocs.io/en/latest/
