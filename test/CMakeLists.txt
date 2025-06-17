cmake_minimum_required(VERSION 3.16)

project(test LANGUAGES C CXX)

include(external.cmake)

set(TEST_FILE 
    test_chat_template.cpp
    # test_std.cpp
    test_model.cpp
		test_util.cpp
)

add_executable(test test_main.cpp)
target_sources(test PRIVATE ${TEST_FILE})
target_link_libraries(test PRIVATE Catch2 common llama)

add_executable(test_cli test_cli.cpp )
target_link_libraries(test_cli PRIVATE Catch2)

add_executable(test_curl test_main.cpp test_curl.cpp)
target_link_libraries(test_curl CURL::libcurl Catch2)

# add_executable(test_curl test_curl.cp)

# add_subdirectory(test_make_installer)
