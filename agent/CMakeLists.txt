cmake_minimum_required(VERSION 3.16)
project(avllm_python)

# Set C++ standard and shared library settings
if(MSVC)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Important for .so
endif()


# cmake export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "For clangd" FORCE)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)

FetchContent_MakeAvailable(pybind11)

add_subdirectory(llama_cpp)

pybind11_add_module(avllm src/avllm_bindings.cpp)
target_link_libraries(avllm PRIVATE avllm_backend)

install(TARGETS avllm
     EXPORT avllmTargets
     RUNTIME DESTINATION avllm
     LIBRARY DESTINATION avllm
     ARCHIVE DESTINATION avllm
)

