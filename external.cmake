include(FetchContent)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/avble/llama.cpp.git
    GIT_TAG        1955993e
)

FetchContent_Declare(
    av_connect
    GIT_REPOSITORY https://github.com/avble/av_connect.git
    GIT_TAG        fc58e2f 
)


FetchContent_GetProperties(llama_cpp)
if(NOT llama_cpp_POPULATED)
  FetchContent_Populate(llama_cpp)
  set(GGML_AMX OFF)
  option(LLAMA_CURL "llama libcur" OFF)
  option(LLAMA_BUILD_SERVER "llama server" OFF)
  option(LLAMA_BUILD_EXAMPLES "llama example" OFF)
  option(LLAMA_BUILD_COMMON "llama: build common utils library" ON)
  option(LLAMA_BUILD_TESTS "llama: build llama tests" OFF)
  add_subdirectory(${llama_cpp_SOURCE_DIR} ${llama_cpp_BINARY_DIR})
endif()

FetchContent_GetProperties(av_connect)
if(NOT av_connect_POPULATED)
  FetchContent_Populate(av_connect)
  option(AV_CONNECT_BUILD_EXAMPLES "av_connect: Build examples" OFF)
  add_subdirectory(${av_connect_SOURCE_DIR} ${av_connect_BINARY_DIR})
endif()

