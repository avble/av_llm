
cmake_minimum_required(VERSION 3.16)

project(av_llm 
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ENABLE_PROGRAMS OFF CACHE BOOL "Disable programs")
set(ENABLE_TESTING OFF CACHE BOOL "Disable tests")


# set(BUILD_SHARED_LIBS OFF)

# if (APPLE)
#     set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
# endif()

include(external.cmake)

option(AV_LLM_BUILD_TEST "av_llm: build test " OFF)

set(TARGET_SRCS
    av_llm.cpp
)

set(PUBLIC_ASSETS
    index.html.gz
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

set(TARGET av_llm)
add_executable(${TARGET} ${TARGET_SRCS})

if(MSVC)
    target_compile_options(${TARGET} PRIVATE /bigobj)
    add_compile_options(/std:c++17)
endif()

target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} av_connect CURL::libcurl )
target_include_directories(${TARGET} PRIVATE ${av_connect_SOURCE_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

file(COPY simplechat DESTINATION  ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(example)

if (AV_LLM_BUILD_TEST STREQUAL "ON")
    add_subdirectory(test)
endif()