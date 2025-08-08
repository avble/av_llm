if(MSVC)
	add_compile_options(/std:c++17)
endif()

add_executable(avllm_semantic_search semantic_search.cpp)
target_link_libraries(avllm_semantic_search PUBLIC llama common)
if(MSVC)
	target_compile_options(avllm_semantic_search PRIVATE /bigobj)
endif()

