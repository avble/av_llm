# Run this before running the tool:
# $ docker image pull python:3.11
import io
import tarfile
from typing import Any, AsyncIterator
import tempfile
import os
import subprocess

import docker
from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)

from ..tool import Tool

_docker_client = None

PYTHON_EXECUTION_BACKEND = "docker"

if os.environ.get("PYTHON_EXECUTION_BACKEND") == "dangerously_use_uv":
    PYTHON_EXECUTION_BACKEND = "dangerously_use_uv"


def call_python_script(script: str) -> str:
    """
    Call a python script by writing it to a file in the container and executing it.
    """
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
        # pull image `python:3.11` if not present
        try:
            _docker_client.images.get("av_python:3.11")
        except docker.errors.ImageNotFound:
            _docker_client.images.pull("av_python:3.11")

    # 1. Create a temporary tar archive containing the script
    script_name = "script.py"
    tarstream = io.BytesIO()
    with tarfile.open(fileobj=tarstream, mode="w") as tar:
        script_bytes = script.encode("utf-8")
        tarinfo = tarfile.TarInfo(name=script_name)
        tarinfo.size = len(script_bytes)
        tar.addfile(tarinfo, io.BytesIO(script_bytes))
    tarstream.seek(0)

    # 2. Start the container
    container = _docker_client.containers.create(
        "av_python:3.11", command="sleep infinity", detach=True
    )
    try:
        container.start()
        # 3. Put the script into the container
        container.put_archive(path="/tmp", data=tarstream.read())
        # 4. Execute the script
        exec_result = container.exec_run(f"python /tmp/{script_name}")

        exec_result = container.exec_run("ls /tmp")
        file_list = exec_result.output.decode("utf-8").splitlines()

        png_files = [f for f in file_list if f.lower().endswith(".png")]

        if png_files:
            filename = png_files[0]  # Pick the first file (or use logic to pick a specific one)

            file_path = f"/tmp/{filename}"
            stream, stat = container.get_archive(file_path)

            file_like = io.BytesIO(b''.join(stream))  # Get the tar stream in memory
            with tarfile.open(fileobj=file_like) as tar:
                tar.extractall(path=".")  # Save the file locally to ./plot.png
            
            # current working directory and append filename
            cwd = os.getcwd()
            cwd = cwd.replace("\\", "/")  # for windows path compatibility:w
            return f"the file is saved at {cwd}/{filename}"

    finally:
        container.remove(force=True)
    return "the result is good"


def call_python_script_with_uv(script: str) -> str:
    """
    Call a python script by writing it to a file to a temporary directory
    and executing it with uv.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(script)
        exec_result = subprocess.run(
            ["uv", "run", "--no-project", "python", script_path],
            capture_output=True)
        return exec_result.stdout.decode("utf-8")


class PythonTool(Tool):
    def __init__(
        self,
        name: str = "python",
    ):
        assert name == "python"

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
#         return """
# Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
# When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you. You have to use print statements to access the output.
#         """.strip()
        return """
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user.
When you send a message containing python code to python, it will be executed in a stateless docker container.
The docker will execute the python code and return the 
+ if the result is in graph, histogram, table, the png file content name, plot_result.png, is saved to /tmp folder
        """.strip()


    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(), description=self.instruction, tools=[]
        )

    def _make_response(
        self,
        output: str,
        channel: str | None = None,
    ) -> Message:
        content = TextContent(text=output)
        return self.make_response(content=content, channel=channel)

    def make_response(
        self,
        content: Content,
        *,
        metadata: dict[str, Any] | None = None,
        author: Author | None = None,
        channel: str | None = None,
    ) -> Message:
        tool_name = self.get_tool_name()
        author = Author(role=Role.TOOL, name=f"{tool_name}")

        message = Message(
            author=author,
            content=[content],
        ).with_recipient("assistant")

        if channel:
            message = message.with_channel(channel)

        return message

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        script = message.content[0].text
        channel = message.channel
        if PYTHON_EXECUTION_BACKEND == "docker":
            output = call_python_script(script)
        elif PYTHON_EXECUTION_BACKEND == "dangerously_use_uv":
            output = call_python_script_with_uv(script)
        else:
            raise ValueError(
                f"Invalid PYTHON_EXECUTION_BACKEND: {PYTHON_EXECUTION_BACKEND}"
            )
        yield self._make_response(output, channel=channel)
