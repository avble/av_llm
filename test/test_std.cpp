#include "catch2/catch.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

TEST_CASE("test_std_01")
{

    auto home_path = std::getenv("HOME") ? std::filesystem::path(std::getenv("HOME")) : std::filesystem::path();
    auto app_path  = home_path / ".av_llm";

    { // create a file
        std::filesystem::create_directories(app_path);
        auto file_path = app_path / "file_01";

        // open a file
        std::ofstream of(file_path.c_str());
        if (!of.is_open())
        {
            std::cerr << "can not open file \n";
            exit(0);
        }

        of << "hello world";

        of.close();
    }

    { // list all files

        for (const auto & entry : std::filesystem::directory_iterator(app_path))
        {
            if (entry.is_regular_file())
                std::cout << entry.path().filename() << std::endl;
        }
    }

    { // delete a files
        std::string file_name = "file_01";
        std::filesystem::remove(app_path / file_name);
    }
}
