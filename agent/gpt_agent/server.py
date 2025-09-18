import argparse

import uvicorn
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
)

from .api_server import create_api_server
from .inference.llama_cpp import setup_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Responses API server")
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
        default="~/model",
        required=False,
    )
    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    args = parser.parse_args()


    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    infer_next_token = setup_model(args.checkpoint)
    uvicorn.run(create_api_server(infer_next_token, encoding), port=args.port)
