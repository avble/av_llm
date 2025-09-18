import os
os.add_dll_directory(r"C:\program files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")

import gradio as gr
from gradio import update
import datetime
from openai import OpenAI, AsyncOpenAI
from openai.types.responses import *
import re
import json
import time

oai_client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="no-need"
)  # Uses API key from env var



async def chat_with_model(message, history):    

    if not message.strip():
        yield history, update(visible=False)
        return 

    chat_msg = history or []
    chat_msg.append([message, ""])
    yield chat_msg, update(visible=False)

    state = 0
        
    try:
        response = await oai_client.responses.create(
            model = "gpt-oss",
            input  = message,
            reasoning ={ "effort": "medium" },
            #tools = [{ "type": "code_interpreter" }, { "type": "browser_search" }],
            tools = [{ "type": "browser_search" }],
            stream=True
        )

        # async receive the stream
        markdown_is_open = False 
        async for event in response: 
            if isinstance(event, ResponseOutputItemAddedEvent):
                print("[output_item]", event.item.type, flush=True)
                if event.item.type == "reasoning" and markdown_is_open == False:
                    #chat_msg[-1][1] += "\n```\n[Reasoning]:\n"
                    #chat_msg[-1][1] += "\n**[Reasoning]**\n"
                    chat_msg[-1][1] += "\n#### 🤔 Reasoning\n"
                    #markdown_is_open = True
                    yield chat_msg, update(value="🤔 Thinking...", visible=True)
                elif event.item.type == "web_search_call":
                    #chat_msg[-1][1] += "\n```\n[web-search...]:\n"
                    #markdown_is_open = True
                    yield chat_msg, update(value="🔍 Searching the web...", visible=True)
                elif event.item.type == "message":
                    chat_msg[-1][1] += "\n\n## 📝 **Final Answer:** \n"
                    yield chat_msg, update(visible=False)
                elif markdown_is_open == True:
                    chat_msg[-1][1] += "\n```"
                    markdown_is_open = False 
                    yield chat_msg, update(visible=False)
            elif isinstance(event, ResponseOutputItemDoneEvent):
                print("[output_item_done]", event.item.type, flush=True)
                if event.item.type == "reasoning" and markdown_is_open == True:
                    #chat_msg[-1][1] += "\n```\n"
                    chat_msg[-1][1] += "\n\n"
                    markdown_is_open = False 
                    yield chat_msg, update(visible=False)
                elif event.item.type == "web_search_call":
                    #if markdown_is_open == True:
                    #    chat_msg[-1][1] += "\n```\n"
                    #markdown_is_open = False 
                    yield chat_msg, update(visible=False)
                elif markdown_is_open == True:
                    chat_msg[-1][1] += "\n```"
                    markdown_is_open = False 
                    yield chat_msg, update(visible=False)

            elif isinstance(event, ResponseContentPartAddedEvent): 
                print("[content_part_added]", event.part.type, flush=True)
            elif isinstance(event, ResponseContentPartDoneEvent):
                print("[content_part_done]", event.part.type, flush=True)
            elif isinstance(event, ResponseTextDeltaEvent):
                #print("[ResponseTextDeltaEvent]", event.delta, flush=True)
                #print(event.delta, flush=True, end="")
                chat_msg[-1][1] += event.delta
                yield chat_msg, update(visible=False)
            elif isinstance(event, ResponseReasoningTextDeltaEvent):
                #print("ResponseReasoningTextDeltaEvent", event.delta, flush=True, end="")
                #print(event.delta, flush=True, end="")
                chat_msg[-1][1] += event.delta
                yield chat_msg, update(visible=True)
            elif isinstance(event, ResponseReasoningTextDoneEvent):
                print(event.type, flush=True)
            elif isinstance(event, ResponseCompletedEvent):
                print("[completed]", flush=True)
                break
            elif isinstance(event, ResponseWebSearchCallInProgressEvent):
                print("[web_search_call_in_progress]", event.item_id, event.sequence_number, flush=True)
            else:
                print(event.type, flush=True)

        print("[chat_with_model] end here")

    except Exception as e:
        error_message = f"❌ Network Error"
        chat_msg[-1][1] = error_message 
        print(str(e))
        yield chat_msg , update(visible=False)
        return

    
# Create the Gradio interface
with gr.Blocks(title="💬 Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot")
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=800)
            status = gr.Markdown("⏳ Waiting...", visible=False)

            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            
            clear_btn = gr.Button("Clear Chat")

        
    # Chat functionality
    inputs = [msg, chatbot]
    outputs = [chatbot, status]
    
    msg.submit(chat_with_model, inputs, outputs)
    send_btn.click(chat_with_model, inputs, outputs)
    clear_btn.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()

