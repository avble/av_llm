#include "CLI11.hpp"

#include <iostream>
#include <regex>
#include <string>

#include "catch2/catch.hpp"

int main(int argc, char * argv[])
{
    const std::regex pattern(R"((run|chat|download)(.*))");

    while (true)
    {
        std::string line;
        printf("\033[32m> \033[0m");
        std::getline(std::cin, line);

        std::smatch match;

        if (std::regex_match(line, match, pattern))
        {
            std::string command;
            command = match[1];

            std::vector<std::string> args;
            std::istringstream iss(match[2]);
            std::string word;
            while (iss >> word)
                args.push_back(word);

            std::cout << "cmd: " << command << std::endl;

            std::cout << "args: \n";

            std::for_each(args.begin(), args.end(), [](const std::string & arg) { std::cout << arg << ","; });
            std::cout << "\n";
        }
        else
            std::cout << "Not match" << std::endl;
    }
}
