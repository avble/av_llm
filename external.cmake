include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/avble/llama.cpp.git
    GIT_TAG        main
)



FetchContent_Declare(
    av_connect
    GIT_REPOSITORY https://github.com/avble/av_connect.git
    GIT_TAG        main
)

if(NOT llama_cpp_POPULATED)
  option(LLAMA_CURL "llama libcur" ON)
  option(LLAMA_BUILD_SERVER "llama server" ON)
  option(LLAMA_BUILD_EXAMPLES "llama example" ON)
  option(LLAMA_BUILD_COMMON "llama: build common utils library" ON)
  FetchContent_Populate(llama_cpp)
  add_subdirectory(${llama_cpp_SOURCE_DIR} ${llama_cpp_BINARY_DIR})
endif()

if(NOT av_connect_POPULATED)
  FetchContent_Populate(av_connect)
  add_subdirectory(${av_connect_SOURCE_DIR} ${av_connect_BINARY_DIR})
endif()

