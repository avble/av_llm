cmake_minimum_required(VERSION 3.16)
project(avllm_python)

# Set C++ standard and shared library settings
if(MSVC)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Important for .so
endif()


set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# === llama_cpp Build Options ===
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_METAL_NDEBUG ON CACHE BOOL "" FORCE)
set(GGML_CUDA          ON CACHE BOOL "" FORCE)

# Optionally disable other things if not needed:
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)


include(FetchContent)

FetchContent_Declare(
	llama_cpp
	GIT_REPOSITORY https://github.com/avble/llama.cpp.git
	GIT_TAG        main #ca493d59 #ecc7e88b #main
)


FetchContent_MakeAvailable(llama_cpp)

# Your C++ backend
add_library(avllm_backend STATIC avllm.cpp)
target_link_libraries(avllm_backend PRIVATE common llama)
target_include_directories(avllm_backend PUBLIC include)


