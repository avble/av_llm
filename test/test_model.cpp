#include "catch2/catch.hpp"

#include "../src/model.hpp"

#include <iomanip>
#include <iostream>

TEST_CASE("test_model_printf")
{

    /*
    | Model Name    | Notes         |
    | ------------- | ------------- |
    | qwen2.5-coder | tools, coding |

    */

    pre_config_model_init();

    std::cout << std::left;

    std::cout << std::setw(40) << "Model Name" << "|" << std::setw(20) << "Remarks" << "|";
    std::cout << '\n' << std::string(40, '-') << "|" << std::string(20, '-') << "|" << '\n';
    for (const auto & model : pre_config_model)
    {
        std::cout << std::setw(40) << model.first << "|" << std::string(20, ' ') << "|" << '\n';
    }
    std::cout << std::endl;
}
