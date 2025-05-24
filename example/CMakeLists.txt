cmake_minimum_required(VERSION 3.16)

project(av_simple
        LANGUAGES CXX C)


function(add_llm_example name)
        add_executable(${name} ${name}.cpp)
        target_link_libraries(${name} PUBLIC llama common)
endfunction()

add_llm_example(avllm_gen)
add_llm_example(avllm_chat)
