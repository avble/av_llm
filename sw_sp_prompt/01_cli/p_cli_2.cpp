#include <CLI/CLI.hpp>
#include <iostream>
#include <string>
#include <csignal>

bool running = true;

void handle_signal(int) {
    static int ctrl_c_count = 0;
    if (++ctrl_c_count >= 2) {
        running = false;
    }
}

int main(int argc, char **argv) {
    CLI::App app{"av_llm - A CLI for LLM management and interaction"};

    // Global options
    int option_1 = 0;
    std::string option_2;
    app.add_option("--option_1", option_1, "This is option 1 (int)");
    app.add_option("--option_2", option_2, "This is option 2 (string)");

    // Global flags
    bool show_version = false;
    app.add_flag("-v,--version", show_version, "Show version");

    // === model command ===
    auto model = app.add_subcommand("model", "Model management commands");

    auto model_ls = model->add_subcommand("ls", "List all models");

    std::string pull_model_name;
    auto model_pull = model->add_subcommand("pull", "Pull a model");
    model_pull->add_option("model_name", pull_model_name, "Model name to pull")->required();

    auto model_del = model->add_subcommand("del", "Delete a model");

    // === chat command ===
    std::string chat_model_path;
    auto chat = app.add_subcommand("chat", "Start interactive chat. Ends with double Ctrl-C.");
    chat->add_option("model_path", chat_model_path, "Model path to use")->required();

    // === serve command ===
    std::string serve_model_path;
    auto serve = app.add_subcommand("serve", "Serve a model as an API endpoint");
    serve->add_option("model_path", serve_model_path, "Model path to serve")->required();

    // Parse input
    CLI11_PARSE(app, argc, argv);

    if (show_version) {
        std::cout << "av_llm version 1.0.0\n";
        return 0;
    }

    if (model->parsed()) {
        if (model_ls->parsed()) {
            std::cout << "Listing models...\n";
        } else if (model_pull->parsed()) {
            std::cout << "Pulling model: " << pull_model_name << "\n";
        } else if (model_del->parsed()) {
            std::cout << "Deleting model...\n";
        } else {
            std::cout << "No valid subcommand for 'model'. Use -h for help.\n";
        }
    } else if (chat->parsed()) {
        std::cout << "Starting interactive chat with model: " << chat_model_path << "\n";
        std::cout << "(Press Ctrl-C twice to quit)\n";
        std::signal(SIGINT, handle_signal);
        std::string input;
        while (running) {
            std::cout << "You> ";
            if (!std::getline(std::cin, input)) break;
            std::cout << "Model> [responds to]: " << input << "\n";
        }
        std::cout << "\nChat ended.\n";
    } else if (serve->parsed()) {
        std::cout << "Serving model at: " << serve_model_path << "\n";
        // In a real app, start server here
    } else if (argc == 1) {
        std::cout << "No command provided. Use -h to see available commands.\n";
    }

    return 0;
}

