cmake_minimum_required(VERSION 3.16)

project(test LANGUAGES C CXX)

include(external.cmake)

set(TEST_FILE test_chat_template.cpp)


add_executable(test test_main.cpp)
target_sources(test PRIVATE ${TEST_FILE})

target_link_libraries(test PRIVATE Catch2 common llama)


# cli application
# set(test_cli_SRCS linenoise-ng/linenoise.cpp linenoise-ng/ConvertUTF.cpp linenoise-ng/wcwidth.cpp)
# message("test ... " ${test_cli_SRCS})
add_executable(test_cli test_cli.cpp )
# target_sources(test_cli PUBLIC ${test_cli_SRCS})
# target_include_directories(test_cli PUBLIC linenoise-ng)
target_link_libraries(test_cli PRIVATE Catch2)