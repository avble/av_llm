{
  "model": "gpt-oss",
  "input": "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-08-19\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>how are you today?<|end|><|start|>assistant",
  "instructions": "restart",
  "tool_choice": "auto",
  "stream": true
}

{
  "model": "gpt-oss",
  "input": "what is the meaning of life",
  "instructions": "restart",
  "tool_choice": "auto",
  "stream": true
}

curl -X POST http://127.0.0.1:8080/v1/responses -H "Content-Type: application/json" -N -d @curl.json


```python
import gpt_agent


text = """<|channel|>analysis<|message|>We need to browse. Let's search.<|end|><|start|>assistant<|channel|>analysis to=browser.search code<|message|>{"query": "Ho Chi Minh City weather today", "topn": 10, "source": "news"}"""
print(gpt_agent.parse_harmony_text(text))
```


```python
import asyncio

from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend

async def main():
    backend = ExaBackend(
        source="web",
    )

    browser_tool = SimpleBrowserTool(backend=backend, max_search_results = 5)

    # search
    print("==================")
    print("search")
    async for msg in browser_tool.search("weather in seoul"):
        for content in msg.content:
            print(content.to_dict().get("text", ""))


    # tool state
    print("==================")
    print("pages")
    for key, page in browser_tool.tool_state.pages.items():
        print("key:", key)
        print("text:", page.text)
        print("urls:", page.urls)

    # open
    print("==================")
    print("open")
    for i in range(4):
        print("open ", i)
        async for msg in browser_tool.open(i, 0):
            for content in msg.content:
                print(content.to_dict().get("text", ""))        


asyncio.run(main())
```
