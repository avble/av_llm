#include <iostream>
#include <string>
#include <vector>
#include <csignal>
#include <atomic>

// Globals for signal handling
std::atomic<int> ctrl_c_count(0);

void signal_handler(int signal) {
    if (signal == SIGINT) {
        ctrl_c_count++;
        if (ctrl_c_count >= 2) {
            std::cout << "\nExiting chat...\n";
            exit(0);
        } else {
            std::cout << "\n(Press Ctrl-C again to exit chat)\n";
        }
    }
}

void print_help() {
    std::cout << R"(av_llm - CLI program

Usage:
  av_llm [--option_1 N] [--option_2 STR] [flags] <command> [param] [sub-command] [sub-param]

Global Options:
  --option_1 N         this is the option 1. type int
  --option_2 STR       this is option 2. type string

Flags:
  -h, --help           Print help
  -v, --version        Show version

Commands:
  model                Model operations
    ls                 List all models
    pull <model>       Pull a model (model: string)
    del <model>        Delete a model

  chat <model-path>    Start an interactive chat with the model (Ctrl-C twice to exit)
  serve <model-path>   Serve model (string param: model-path)
)" << std::endl;
}

void print_version() {
    std::cout << "av_llm version 0.1.0" << std::endl;
}

void model_ls() {
    std::cout << "[Model] Listing all models..." << std::endl;
    // TODO: Implement listing logic
}

void model_pull(const std::string& model) {
    std::cout << "[Model] Pulling model: " << model << std::endl;
    // TODO: Implement pull logic
}

void model_del(const std::string& model) {
    std::cout << "[Model] Deleting model: " << model << std::endl;
    // TODO: Implement delete logic
}

void chat(const std::string& model_path) {
    std::cout << "[Chat] Starting chat with model at: " << model_path << std::endl;
    std::cout << "(Press Ctrl-C twice to exit chat)" << std::endl;
    std::signal(SIGINT, signal_handler);

    std::string input;
    while (true) {
        ctrl_c_count = 0;
        std::cout << "You: ";
        if (!std::getline(std::cin, input)) {
            std::cout << "\nEnd of input detected. Exiting chat.\n";
            break;
        }
        // TODO: Implement chat logic
        std::cout << "[Model] (echo) " << input << std::endl;
    }
}

void serve(const std::string& model_path) {
    std::cout << "[Serve] Serving model at: " << model_path << std::endl;
    // TODO: Implement serve logic
}

int main(int argc, char* argv[]) {
    int option_1 = 0;
    std::string option_2;
    bool help_flag = false, version_flag = false;

    std::vector<std::string> args(argv + 1, argv + argc);
    size_t idx = 0;

    // Parse global options and flags
    while (idx < args.size()) {
        if (args[idx] == "-h" || args[idx] == "--help") {
            help_flag = true;
            idx++;
        } else if (args[idx] == "-v" || args[idx] == "--version") {
            version_flag = true;
            idx++;
        } else if (args[idx] == "--option_1" && idx + 1 < args.size()) {
            option_1 = std::stoi(args[idx + 1]);
            idx += 2;
        } else if (args[idx] == "--option_2" && idx + 1 < args.size()) {
            option_2 = args[idx + 1];
            idx += 2;
        } else {
            break;
        }
    }

    if (help_flag) {
        print_help();
        return 0;
    }
    if (version_flag) {
        print_version();
        return 0;
    }

    // Parse command
    if (idx >= args.size()) {
        print_help();
        return 1;
    }

    std::string command = args[idx++];
    if (command == "model") {
        if (idx >= args.size()) {
            std::cerr << "Missing model subcommand\n";
            print_help();
            return 1;
        }
        std::string subcmd = args[idx++];
        if (subcmd == "ls") {
            model_ls();
        } else if (subcmd == "pull" && idx < args.size()) {
            model_pull(args[idx]);
        } else if (subcmd == "del" && idx < args.size()) {
            model_del(args[idx]);
        } else {
            std::cerr << "Unknown or incomplete model subcommand\n";
            print_help();
            return 1;
        }
    } else if (command == "chat") {
        if (idx < args.size()) {
            chat(args[idx]);
        } else {
            std::cerr << "chat command requires <model-path> parameter\n";
            print_help();
            return 1;
        }
    } else if (command == "serve") {
        if (idx < args.size()) {
            serve(args[idx]);
        } else {
            std::cerr << "serve command requires <model-path> parameter\n";
            print_help();
            return 1;
        }
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_help();
        return 1;
    }
    return 0;
}
