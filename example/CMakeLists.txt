cmake_minimum_required(VERSION 3.19)

if(MSVC)
	add_compile_options(/std:c++17)
endif()

function(add_llm_example name)
	add_executable(${name} ${name}.cpp)
	target_link_libraries(${name} PUBLIC llama common)
	if(MSVC)
		target_compile_options(${name} PRIVATE /bigobj)
	endif()
endfunction()

add_llm_example(avllm_gen)
add_llm_example(avllm_chat)
add_llm_example(avllm_chat_ng)
add_llm_example(avllm_cli)
add_llm_example(avllm_embedding)
add_llm_example(avllm_sentence_similarity)


add_subdirectory(semantic_search)

#add_executable(llama_embedding llama_embedding.cpp) 
#target_link_libraries(llama_embedding common llama)
# add_executable(avllm_server avllm_server.cpp)
# if (MSVC)
# 	target_compile_options(avllm_server PRIVATE /bigobj)
# endif()
# 
# target_link_libraries(avllm_server PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} av_connect)
# target_include_directories(avllm_server PRIVATE ${av_connect_SOURCE_DIR}/lib ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR} )
