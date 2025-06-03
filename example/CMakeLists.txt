cmake_minimum_required(VERSION 3.16)

project(av_simple
        LANGUAGES CXX C)


function(add_llm_example name)
        add_executable(${name} ${name}.cpp)
        target_link_libraries(${name} PUBLIC llama common)
endfunction()

add_llm_example(avllm_gen)
add_llm_example(avllm_chat)
add_llm_example(avllm_server)

set(PUBLIC_ASSETS
    ../index.html.gz
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)

endforeach()

target_link_libraries(avllm_server PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} av_connect)
# message("message-av_connect: " ${av_connect_SOURCE_DIR})
target_include_directories(avllm_server PRIVATE ${av_connect_SOURCE_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(avllm_server PRIVATE cxx_std_17)