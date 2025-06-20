
cmake_minimum_required(VERSION 3.17)

project(av_llm 
    LANGUAGES CXX C)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(AV_LLM_BUILD_TEST "av_llm: build test " OFF)
option(AV_LLM_USE_SYSTEM_CURL "av_llm: using system lib cur" OFF)

include("external.cmake")

set(PUBLIC_WEBUI
    index.html.gz
)

set(PUBLIC_WEBUI_SRCS )

foreach(asset ${PUBLIC_WEBUI})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
		list(APPEND PUBLIC_WEBUI_SRCS ${output})
    add_custom_command(
        OUTPUT "${output}"
        DEPENDS "${input}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)

endforeach()

set(TARGET_SRCS
    av_llm.cpp
)

set(TARGET av_llm)
add_executable(${TARGET} ${TARGET_SRCS} ${PUBLIC_WEBUI_SRCS})

if(MSVC)
    target_compile_options(${TARGET} PRIVATE /bigobj)
    add_compile_options(/std:c++17)
endif()

if (AV_LLM_USE_SYSTEM_CURL STREQUAL "ON")
	find_package(CURL REQUIRED)
	message(STATUS "libcurl is found: ${CURL_VERSION_STRING}")
endif()

target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} av_connect CURL::libcurl CLI11::CLI11 )
target_include_directories(${TARGET} PRIVATE ${av_connect_SOURCE_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_subdirectory(example)

#if (AV_LLM_BUILD_TEST STREQUAL "ON")
    add_subdirectory(test)
		#endif()
