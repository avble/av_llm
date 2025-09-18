from skbuild import setup

setup(
    name="avllm",
    version="0.1.0",
    description="Python bindings for AV LLM C++ backend",
    author="Your Name",
    packages=["avllm"],
    package_dir={"": "src"}, 
    include_package_data=True,
    cmake_install_dir="avllm",
    python_requires=">=3.7",
)

