cmake_minimum_required(VERSION 3.14)
project(av_llm_cli
    VERSION 0.1.0
    DESCRIPTION "av_llm - CLI interface using CLI11"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CLI11
# If you have CLI11 installed system-wide, you can use find_package(CLI11 CONFIG REQUIRED)
# Otherwise, fetch it automatically:
include(FetchContent)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(CLI11)

function(add_demo name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE CLI11::CLI11)
    # Optional: install target
    install(TARGETS ${name} RUNTIME DESTINATION bin)
endfunction()

add_demo(p_cli_1)
add_demo(p_cli_2)
add_demo(p_cli_3)

# add_executable(av_llm main.cpp)
# target_link_libraries(av_llm PRIVATE CLI11::CLI11)


# Optional: generate compile_commands.json for LSP/autocomplete
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
