cmake_minimum_required(VERSION 3.12)

# Set the project name and version
project(Avllm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")

add_executable(av_llm_installer src/main.cpp)


