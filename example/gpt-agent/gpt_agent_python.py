"""
+ Demo the agentic `python code execution`
"""


import gradio as gr
import datetime
from openai import OpenAI
import re
import json
import time
from openai_harmony import SystemContent, Message, Conversation, Role, load_harmony_encoding, HarmonyEncodingName
from gpt_oss.tools.python_docker.docker_tool import PythonTool

oai_client = OpenAI(
    base_url="http://127.0.0.1:8081/v1",
    api_key="no-need"
)  # Uses API key from env var

total_request_time = 0.0  # Total accumulated request/response time in seconds
total_input_tokens = 0
total_output_tokens = 0

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
python_tool = PythonTool()



def parse_harmony_text(raw_text: str):
    """
    i.e.
    input: 
    ~~~~~~
    <|channel|>analysis<|message|>We respond.<|end|><|start|>assistant<|channel|>final<|message|>Hi! How can I help you today?

    output: 
    ~~~~~~
    {'analysis': 'We respond.', 'final': 'Hi! How can I help you today?'}
    
    """    
    pattern = r"(?:<\|start\|>)?(?:assistant)?<\|channel\|>(.*?)<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|$)"

    matches = re.findall(pattern, raw_text, re.DOTALL)    
    parsed = {}
    for channel, content in matches:
        parsed[channel.strip()] = content.strip()
    
    return parsed




def add_static_log(total_time: int, llm_time: int) -> str:
    
    ts = total_input_tokens + total_output_tokens
    ts = int((ts/total_time)*1000)
    log = {}

    log["time"] = {
    "total": f"{total_time:04d}",
    "decoding": f"{llm_time:04d}"
    }
    log["token"] = {

    }

    log = {
    "time": {
        "total": total_time,
        "decoding": llm_time
    },
    "tokens": {
        "token_per_sec": ts,
        "input": total_input_tokens,
        "decoding": total_output_tokens
    }
    }

    log_summary = (
    f"time(ms):   {log['time']}\n"
    f"tokens:     {log['tokens']}"
    )    
    return f"\n```python\n{log_summary}\n```"

async def chat_with_model(message, history):    
    global total_input_tokens
    global total_output_tokens

    if not message.strip():
        return history, ""
        
    # construct the system message
    system_message_content = SystemContent.new().with_conversation_start_date(
        datetime.datetime.now().strftime("%Y-%m-%d")
    )

    total_request_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    # First request
    start_time_time = time.time()
    chat_messages = []
    system_message_content = system_message_content.with_tools(python_tool.tool_config)

    # Append user message and empty assistant placeholder (idiomatic Gradio pattern)
    history = history + [[message, ""]] 
    chat_messages.append(Message.from_role_and_content(Role.USER, message))
        
    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    # create the overall prompt
    messages = [system_message] + chat_messages

    conversation = Conversation.from_messages(messages)

    # convert to tokens
    token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    harmony_msg_turn_1 = encoding.decode_utf8(token_ids)
    
    try:
        start_time = time.time()
        response = oai_client.responses.create(
            model = "gpt-oss",
            input  = harmony_msg_turn_1,
            instructions="restart",
            tool_choice="auto",
            extra_body={"play": "restart"},
            stream=False
        )
        
        # Extract assistant text
        harmony_msg = ""        
        for block in response.output[0].content:
            if block.type == "output_text":
                harmony_msg += block.text
        end_time = time.time()
        total_request_time += (end_time - start_time)

        usage = response.usage
        total_input_tokens = usage.input_tokens
        total_output_tokens = usage.output_tokens
            
        print("harmony_msg")
        print(harmony_msg)

        harmony_text = parse_harmony_text(harmony_msg)
        if (msg := harmony_text.get("final", "")):
            time_time = time.time()
            total_total_time = 0.0
            total_total_time = time_time - start_time_time
            history[-1] = [message, msg + "\n" + add_static_log(int(total_total_time*1000), int(total_request_time*1000))]
            return history, "example.png"
        elif (msg := harmony_text.get("commentary to=python code", "")):
            time_time = time.time()
            total_total_time = 0.0
            total_total_time = time_time - start_time_time

            history[-1] = [message, f"```python\n{msg} \n```\n" + add_static_log(int(total_total_time*1000), int(total_request_time*1000))]

            print(history[-1][1])

            message_ = Message.from_role_and_content(Role.ASSISTANT, msg)
            result_image = ""
            async for msg in python_tool.process(message_):
                for cnt_ in msg.content:
                    result_image += cnt_.to_dict().get("text")
                    print(result_image)
            
            return history, result_image
        else:
            raise "Error"

    except Exception as e:
        time_time = time.time()
        total_total_time = 0.0
        total_total_time = time_time - start_time_time
        error_message = f"❌ Network Error"
        history[-1][1] = error_message + "\n" + add_static_log(int(total_total_time*1000), int(total_request_time*1000))
        print(str(e))
        return history, ""


import base64
with open("example.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    base64_string = encoded_string.decode('utf-8')  # Convert bytes to string

    
# Create the Gradio interface
with gr.Blocks(title="💬 Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot")
    
    with gr.Row():
        with gr.Column(scale=1.5):
            chatbot = gr.Chatbot(height=600)
            
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            
            clear_btn = gr.Button("Clear Chat")
        
        with gr.Column(scale=1):
            img = gr.Image(type="filepath", label="Image Preview")    

    # Chat functionality
    inputs = [msg, chatbot]
    outputs = [chatbot, img]
    
    msg.submit(chat_with_model, inputs, outputs)
    send_btn.click(chat_with_model, inputs, outputs)
    clear_btn.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()


"""
Example user prompt: 
+ plot diagram of sine

"""