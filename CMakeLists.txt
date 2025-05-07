
cmake_minimum_required(VERSION 3.16)

project(av_llm 
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(external.cmake)

add_subdirectory(av_connect)

# find_package(PkgConfig)
# pkg_check_modules(http_parser REQUIRED)
# list(APPEND av_connect_dependent_libs  http_parser)

set(TARGET av_llm)
add_executable(${TARGET} main.cpp)

# target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} ${av_connect_SOURCE_DIR}/example/libav_connect.a http_parser)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} av_connect)
target_include_directories(${TARGET} PRIVATE ${av_connect_SOURCE_DIR}/lib )
# target_compile_features(${TARGET} PRIVATE cxx_std_17)

file(COPY simplechat DESTINATION  ${CMAKE_CURRENT_BINARY_DIR})

# add_subdirectory(examples)