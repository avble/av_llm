#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <ciso646>
#endif

using namespace std;

int main(int argc, char * argv[])
{
    if (true)
    {
        common_params common_param;
        auto llama_example_to_str = [](llama_example ex) -> std::string {
            switch (ex)
            {
            case LLAMA_EXAMPLE_COMMON:
                return "LLAMA_EXAMPLE_COMMON";
            case LLAMA_EXAMPLE_SPECULATIVE:
                return "LLAMA_EXAMPLE_SPECULATIVE";
            case LLAMA_EXAMPLE_MAIN:
                return "LLAMA_EXAMPLE_MAIN";
            case LLAMA_EXAMPLE_EMBEDDING:
                return "LLAMA_EXAMPLE_EMBEDDING";
            case LLAMA_EXAMPLE_PERPLEXITY:
                return "LLAMA_EXAMPLE_PERPLEXITY";
            case LLAMA_EXAMPLE_RETRIEVAL:
                return "LLAMA_EXAMPLE_RETRIEVAL";
            case LLAMA_EXAMPLE_PASSKEY:
                return "LLAMA_EXAMPLE_PASSKEY";
            case LLAMA_EXAMPLE_IMATRIX:
                return "LLAMA_EXAMPLE_IMATRIX";
            case LLAMA_EXAMPLE_BENCH:
                return "LLAMA_EXAMPLE_BENCH";
            case LLAMA_EXAMPLE_SERVER:
                return "LLAMA_EXAMPLE_SERVER";
            case LLAMA_EXAMPLE_CVECTOR_GENERATOR:
                return "LLAMA_EXAMPLE_CVECTOR_GENERATOR";
            case LLAMA_EXAMPLE_EXPORT_LORA:
                return "LLAMA_EXAMPLE_EXPORT_LORA";
            case LLAMA_EXAMPLE_MTMD:
                return "LLAMA_EXAMPLE_MTMD";
            case LLAMA_EXAMPLE_LOOKUP:
                return "LLAMA_EXAMPLE_LOOKUP";
            case LLAMA_EXAMPLE_PARALLEL:
                return "LLAMA_EXAMPLE_PARALLEL";
            case LLAMA_EXAMPLE_TTS:
                return "LLAMA_EXAMPLE_TTS";
            case LLAMA_EXAMPLE_COUNT:
                return "LLAMA_EXAMPLE_COUNT";
            default:
                return "UNKNOWN";
            }
        };

        // prepare data
        std::map<std::string, int> data;

        for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++)
        {

            // std::cout << "\x1b[31m" << llama_example_to_str(static_cast<llama_example>(ex)).c_str() << "\x1b[0m\n";
            auto ctx_arg = common_params_parser_init(common_param, (enum llama_example) ex);
            {
                // print all option of ctx arg
                for (const auto & opt : ctx_arg.options)
                    for (const auto & arg : opt.args)
                    {
                        // change to green color
                        // std::cout << "\x1b[32m" << "\t" << arg << "\x1b[0m" << std::endl;
                        data[arg] = data[arg] | (1 << static_cast<int>(ex));
                    }
            }
        }

        // Helper: center text in fixed width
        static auto centerText = [](const string & text, int width) {
            int padding = width - text.size();
            int left    = padding / 2;
            int right   = padding - left;
            return string(left, ' ') + text + string(right, ' ');
        };

        static auto print_text_center = [](const char * text, int width) {
            int padding = width - strlen(text);
            int left    = padding / 2;
            int right   = padding - left;
            printf("%*s%s%*s", left, "", text, right, "");
        };

        // Helper: print row separator line
        // void printSeparator(int columns, int cell_width)

        if (true)
        {
            int row_count    = data.size();
            int column_count = 0;
            if (!data.empty())
                column_count = static_cast<int>(LLAMA_EXAMPLE_COUNT);

            // string, remove the prefix
            auto remove_prefix_LLAMA_EXAMPLE = [](std::string str) -> std::string {
                if (str.find("LLAMA_EXAMPLE_") == 0)
                    str.erase(0, 14);
                return str;
            };

            std::array<int, LLAMA_EXAMPLE_COUNT> cell_widths;
            cell_widths[0] = 30;
            for (int i = 0; i < LLAMA_EXAMPLE_COUNT; i++)
            {
                std::string str    = llama_example_to_str(static_cast<llama_example>(i));
                str                = remove_prefix_LLAMA_EXAMPLE(str);
                cell_widths[i + 1] = str.size();
            }

            static auto printSeparator = [&cell_widths](int columns) {
                cout << "|";
                for (int i = 0; i < columns + 1; ++i)
                { // +1 for row header
                    cout << string(cell_widths[i], '-') << "|";
                }
                cout << endl;
            };

            // Print header (first row)
            printf("|");
            print_text_center("", cell_widths[0]);
            for (int i = 0; i < LLAMA_EXAMPLE_COUNT; i++)
            {
                std::string str = llama_example_to_str(static_cast<llama_example>(i));
                str             = remove_prefix_LLAMA_EXAMPLE(str);
                cout << "|" << centerText(str, cell_widths[i + 1]);
            }
            cout << "|\n";

            printSeparator(column_count);

            // For each column, print the column name and v's from each row
            for (auto row : data)
            {
                printf("|");
                print_text_center(row.first.c_str(), cell_widths[0]);
                for (int col = 0; col < static_cast<int>(LLAMA_EXAMPLE_COUNT); col++)
                {
                    string cell = ((row.second & (1 << col)) != 0) ? "v" : "";
                    printf("|");
                    print_text_center(cell.c_str(), cell_widths[col + 1]);
                }
                cout << "|\n";
                printSeparator(column_count);
            }
        }
    }
}
