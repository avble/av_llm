import gradio as gr
import datetime
from openai import OpenAI
import re
import json
import time
from openai_harmony import SystemContent, Message, Conversation, Role, load_harmony_encoding, HarmonyEncodingName
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend

client = OpenAI(
    base_url="http://127.0.0.1:8081/v1",
    api_key="no-need"
)  # Uses API key from env var

backend = ExaBackend(
    source="web",
)

browser_tool = SimpleBrowserTool(backend=backend, max_search_results = 5)
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def parse_harmony_text(raw_text: str):
    """
    i.e.
    input: <|channel|>analysis<|message|>We respond.<|end|><|start|>assistant<|channel|>final<|message|>Hi! How can I help you today?
    output: {'analysis': 'We respond.', 'final': 'Hi! How can I help you today?'}

    i.e.
    input: <|channel|>analysis<|message|>We need to browse. Let's search.<|end|><|start|>assistant<|channel|>analysis to=browser.search code<|message|>{"query": "Ho Chi Minh City weather today", "topn": 10, "source": "news"}
    output: {'analysis': "We need to browse. Let's search.", 'analysis to=browser.search code': '{"query": "Ho Chi Minh City weather today", "topn": 10, "source": "news"}'}

    
    """
    pattern = r"(?:<\|start\|>)?(?:assistant)?<\|channel\|>(.*?)<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|$)"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    
    parsed = {}
    for channel, content in matches:
        parsed[channel.strip()] = content.strip()
    
    return parsed




total_request_time = 0.0  # Total accumulated request/response time in seconds
total_input_tokens = 0
total_output_tokens = 0

async def chat_with_model(message, history, use_browser_search):
    
    if not message.strip():
        return history, ""
    
    # Append user message and empty assistant placeholder (idiomatic Gradio pattern)
    history = history + [[message, ""]]    
    
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

    if use_browser_search:
        # enables the tool
        system_message_content = system_message_content.with_tools(browser_tool.tool_config)
        # alternatively you could use the following if your tool is not stateless
        system_message_content = system_message_content.with_browser_tool()
        chat_messages.append(Message.from_role_and_content(Role.USER, message))

    if not use_browser_search:
        for user_msg, assistant_msg in history:
            if user_msg:
                chat_messages.append(Message.from_role_and_content(Role.USER, user_msg))

            if assistant_msg:
                chat_messages.append(Message.from_role_and_content(Role.ASSISTANT, assistant_msg))
        

    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    # create the overall prompt
    messages = [system_message] + chat_messages

    conversation = Conversation.from_messages(messages)

    # convert to tokens
    token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    harmony_msg_turn_1 = encoding.decode_utf8(token_ids)
    
    try:
        start_time = time.time()
        response = client.responses.create(
            model = "gpt-oss",
            input  = harmony_msg_turn_1,
            instructions="restart",
            tool_choice="auto",
            extra_body={"play": "restart"},
            stream=False
        )
        
        # Extract assistant text
        harmony_msg_turn_2 = ""        
        for block in response.output[0].content:
            if block.type == "output_text":
                harmony_msg_turn_2 += block.text
        end_time = time.time()
        total_request_time += (end_time - start_time)

        usage = response.usage
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

        # Iterate over the streamed events
        # for event in response:
        #     if event.type == "response.output_text.delta":
        #         harmony_msg_turn_2 += event.delta  # This is a chunk of text
        #         print(event.delta, end="", flush=True)  # Optional: live output
            
        print("harmony_msg_turn_2")
        print(harmony_msg_turn_2)

        def add_static_log(total_time: int, llm_time: int) -> str:
            ts = total_input_tokens + total_output_tokens
            ts = int((ts/total_time)*1000)
            return f"\n\n\n\n ~~~~~\ntime(ms): (total: {total_time:04d}, decoding: {llm_time:04d}) \ntokens: ({ts:4d} tokens/second,  input: {total_input_tokens}, decoding: {total_output_tokens})"
        

        harmony_text = parse_harmony_text(harmony_msg_turn_2)
        if use_browser_search:
            async def browser_open(open_func_desc: str) -> str:
                print(">>>>browser_open[start]")
                nonlocal total_request_time
                nonlocal total_input_tokens
                nonlocal total_output_tokens                
                print(open_func_desc)

                open_data = json.loads(open_func_desc)
                cursor = open_data.get("cursor", -1)
                id = open_data.get("id", -1)

                open_messages = []
                async for msg in browser_tool.open(id, cursor):
                    open_messages.append(msg)                            

                conv = Conversation.from_messages(open_messages)
                
                token_ids = encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
                harmony_msg_turn_5 = encoding.decode_utf8(token_ids)

                print("harmony_msg_turn_5")
                print(harmony_msg_turn_5)

                start_time = time.time()
                response = client.responses.create(
                    model = "gpt-oss",
                    input  = harmony_msg_turn_5,
                    tool_choice="auto",
                    extra_body={"stops": "}"},
                    stream=False
                )

                harmony_msg_turn_6 = ""
                for block in response.output[0].content:
                    if block.type == "output_text":
                        harmony_msg_turn_6 += block.text

                end_time = time.time()
                total_request_time += (end_time - start_time)

                usage = response.usage
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens

                print(harmony_msg_turn_6)
                print("<<<<browser_open[end]")
                return harmony_msg_turn_6
            
            async def browser_find(find_func_desc: str):
                print(">>>>browser_find[start]")
                nonlocal total_request_time
                nonlocal total_input_tokens
                nonlocal total_output_tokens                
                print(find_func_desc)

                print("[analysis to=browser.find code]")
                find_data = json.loads(find_func_desc)
                cursor = find_data.get("cursor", -1)
                pattern= find_data.get("pattern", "")
                find_messages = []
                async for msg in browser_tool.find(pattern, cursor):
                    find_messages.append(msg)

                conv = Conversation.from_messages(find_messages)
                token_ids = encoding.render_conversation_for_completion(conv, Role.ASSISTANT)

                harmony_msg_turn_7 = encoding.decode_utf8(token_ids)

                print("harmony_msg_turn_7")
                print(harmony_msg_turn_7)

                start_time = time.time()
                response = client.responses.create(
                    model = "gpt-oss",
                    input  = harmony_msg_turn_7,
                    tool_choice="auto",
                    extra_body={"stops": "}"},
                    stream=False
                )

                harmony_msg_turn_8 = ""
                for block in response.output[0].content:
                    if block.type == "output_text":
                        harmony_msg_turn_8 += block.text

                end_time = time.time()
                total_request_time += (end_time - start_time)
                
                usage = response.usage
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens


                print("harmony_msg_turn_8")
                print(harmony_msg_turn_8)
                print("<<<<browser_find[end]")
                return harmony_msg_turn_8
            
            async def browser_search(search_func_desc: str):
                # {"query": "current temperature in Korea August 18 2025", "topn": 10, "source": "news"}'}
                print(">>>>browser_find[start]")
                print(search_func_desc)
                nonlocal total_request_time
                nonlocal total_input_tokens
                nonlocal total_output_tokens
                print("[analysis to=browser.search code]")
                search_data = json.loads(search_func_desc)
                query = search_data.get("query", -1)
                source = "news"
                search_messages = []
                async for msg in browser_tool.search(query, 5, 5, source):
                    search_messages.append(msg)

                conv = Conversation.from_messages(search_messages)
                token_ids = encoding.render_conversation_for_completion(conv, Role.ASSISTANT)

                harmony_msg_search = encoding.decode_utf8(token_ids)
                print("harmony_msg_search")
                print(harmony_msg_search)

                start_time = time.time()
                response = client.responses.create(
                    model = "gpt-oss",
                    input  = harmony_msg_search,
                    tool_choice="auto",
                    extra_body={"stops": "}"},
                    stream=False
                )

                harmony_msg_search_res = ""
                for block in response.output[0].content:
                    if block.type == "output_text":
                        harmony_msg_search_res += block.text

                end_time = time.time()
                total_request_time += (end_time - start_time)

                usage = response.usage
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens

                print("harmony_msg_search_res")
                print(harmony_msg_search_res)
                print("<<<<browser_search[end]")
                return harmony_msg_search_res

            cnt_open = 0
            # harmony_text = parse_harmony_text(harmony_msg_turn_4)
            while True:
                # check if final message is found
                final = harmony_text.get("final", "")
                if final:
                    time_time = time.time()
                    total_total_time = 0.0
                    total_total_time = time_time - start_time_time                    
                    history[-1][1] = final  + "\n" + add_static_log(int(total_total_time*1000), int(total_request_time*1000))
                    return history, ""
                elif (open_func_desc := harmony_text.get("analysis to=browser.open code", "")):
                    open_res = await browser_open(open_func_desc)
                    harmony_text = parse_harmony_text(open_res)                        
                    print(f"{cnt_open} browser.open")
                    cnt_open += 1
                elif (find_func_desc := harmony_text.get("analysis to=browser.find code", "")):
                    find_res = await browser_find(find_func_desc)
                    harmony_text = parse_harmony_text(find_res)
                    print(f"{cnt_open} brow.find")
                elif (search_func_desc := harmony_text.get("analysis to=browser.search code", "")):
                    search_res = await browser_search(search_func_desc)
                    harmony_text = parse_harmony_text(search_res)
                    print(f"{cnt_open} brow.search")                            
                # browser_search
                else:
                    raise Exception("Network Error")
                
                print(harmony_text)

        else:
            time_time = time.time()
            total_total_time = 0.0
            total_total_time = time_time - start_time_time
            history[-1] = [message, harmony_text.get("final", "") + "\n" + add_static_log(int(total_total_time*1000), int(total_request_time*1000))]
            return history, ""

    except Exception as e:
        error_message = f"❌ Network Error"
        history[-1][1] = error_message + "\n" + add_static_log(int(total_total_time*1000), int(total_request_time*1000))
        print(str(e))
        return history, ""


# Create the Gradio interface
with gr.Blocks(title="💬 Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=400)
            
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            
            clear_btn = gr.Button("Clear Chat")
        
        with gr.Column(scale=1, min_width=200):
            # Conditional browser search (matching Streamlit logic)
            # In Streamlit: if "show_browser" in st.query_params:
            # For Gradio, we'll always show it (simplified)
            gr.Markdown("#### Built-in Tools") 
            use_browser_search = gr.Checkbox(label="Web search", value=False)                
   
    # Chat functionality
    inputs = [msg, chatbot, use_browser_search]
    outputs = [chatbot, msg]
    
    msg.submit(chat_with_model, inputs, outputs)
    send_btn.click(chat_with_model, inputs, outputs)
    clear_btn.click(lambda: [], outputs=chatbot)


if __name__ == "__main__":
    demo.launch()