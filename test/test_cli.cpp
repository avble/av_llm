
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include "catch2/catch.hpp"

int main(int argc, char * argv[])
{
    const std::regex pattern(R"((run|chat|model)(.*))");

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

            if (command == "model")
            {
                std::smatch smatch_model;
                std::regex pattern_model(R"((ls|download|del)(.*))");
                std::string model_cmd = match[2];
                // remove leading space
                {
                    int i = 0;
                    while (i < model_cmd.size() && std::isspace(static_cast<unsigned char>(model_cmd[i])))
                        i++;

                    model_cmd.erase(0, i);
                }

                if (std::regex_match(model_cmd, smatch_model, pattern_model))
                {
                    std::cout << "sub-command: " << smatch_model[1] << std::endl;
                    std::cout << "with args: " << smatch_model[2] << std::endl;
                }
                else
                    std::cout << "model: invalid \n";
            }
            else
            {
                std::cout << "args: \n";
                std::for_each(args.begin(), args.end(), [](const std::string & arg) { std::cout << arg << ","; });
                std::cout << "\n";
            }
        }
        else
            std::cout << "Not match" << std::endl;
    }
}
