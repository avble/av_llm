import os
os.add_dll_directory(r"C:\program files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")

import json
import threading
import time
from typing import Callable, Optional

import requests
import avllm

from openai_harmony import HarmonyEncodingName, load_harmony_encoding
from openai_harmony import (SystemContent, Message, Conversation, Role)

EOS_TOKEN = 200002  # only used on hard timeout
EOS_TOKEN_1 = 0xeed2 # only used on hard timeout


condition = threading.Condition()
token_buffer: list[int] = []
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

def _start_request(token_ids: list[int], temperature: float):
    print(f"Starting new request with temperature {temperature} and token-size {len(token_ids)}")
    def run():
        print("Token generation thread started.")

        generated_tokens = 0
        end_token_id = encoding.encode("<|endoftext|>", allowed_special="all")[0]
        while True:
            with condition:
                token = avllm.get_next_token()
                generated_tokens += 1
                if generated_tokens > 5000:
                    print("Reached maximum token limit, stopping generation.")
                    token_buffer.append(EOS_TOKEN)
                    condition.notify_all()
                    break

                if token == -1 or token == EOS_TOKEN_1:
                    print("End of sequence token received, stopping generation.", token)
                    token_buffer.append(EOS_TOKEN)
                    condition.notify_all()
                    break


                if token == end_token_id:
                    print("End token generated, stopping generation.")
                    token_buffer.append(EOS_TOKEN)
                    condition.notify_all()
                    break

                token_buffer.append(token)
                condition.notify_all()
                #print(encoding.decode_utf8([token]), end='', flush=True)

        print("Token generation thread ending.")


    avllm.set_prompt(token_ids)
    t = threading.Thread(target=run, name="-stream", daemon=True)
    t.start()
    return t

def infer_next_token(
    tokens: list[int], temperature: float = 0.0, new_request: bool = False
) -> int:

    if new_request:
        _stream_thread = _start_request(token_ids=tokens, temperature=temperature)

    with condition:
        #print("Waiting for token...")
        condition.wait_for(lambda: len(token_buffer) > 0)
        tok = token_buffer.pop(0)
        #print("Got token:", tok)
        return tok

def setup_model(gguf_file: str) -> Callable[[list[int], float, bool], int]:
    avllm.init(gguf_file)
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return infer_next_token
