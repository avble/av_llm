#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

void print_version() {
    std::cout << "av_llm version 0.1.0" << std::endl;
}

int main(int argc, char** argv) {
    CLI::App app{"av_llm - CLI program"};

    // Global options
    int option_1 = 0;
    std::string option_2;
    app.add_option("--option_1", option_1, "this is the option 1. type int");
    app.add_option("--option_2", option_2, "this is option 2. type string");

    // Version flag
    bool version_flag = false;
    app.add_flag("-v,--version", version_flag, "show version");

    // model command group
    auto model = app.add_subcommand("model", "Model operations");

    // model ls
    bool model_ls_flag = false;
    model->add_flag("--ls", model_ls_flag, "List all models");

    // model pull
    std::string pull_model;
    model->add_option("--pull", pull_model, "Pull a model (model: string)");

    // model del
    std::string del_model;
    model->add_option("--del", del_model, "Delete a model (model: string)");

    // chat command
    std::string chat_model_path;
    auto chat = app.add_subcommand("chat", "Start an interactive chat. Get input from std::in. The chat is completed until the user presses ctrl-C twice");
    chat->add_option("model-path", chat_model_path, "Model path")->required();

    // serve command
    std::string serve_model_path;
    auto serve = app.add_subcommand("serve", "Serve model");
    serve->add_option("model-path", serve_model_path, "Model path")->required();

    CLI11_PARSE(app, argc, argv);

    // Version flag
    if (version_flag) {
        print_version();
        return 0;
    }

    // model command logic
    if (*model) {
        if (model_ls_flag) {
            std::cout << "[Model] Listing all models..." << std::endl;
        } else if (!pull_model.empty()) {
            std::cout << "[Model] Pulling model: " << pull_model << std::endl;
        } else if (!del_model.empty()) {
            std::cout << "[Model] Deleting model: " << del_model << std::endl;
        } else {
            std::cerr << "Specify a model subcommand: --ls, --pull <model>, or --del <model>\n";
        }
        return 0;
    }

    // chat command logic
    if (*chat) {
        std::cout << "[Chat] Starting interactive chat with model at: " << chat_model_path << std::endl;
        std::cout << "(Press Ctrl-C twice to exit chat)" << std::endl;
        // TODO: implement chat logic with Ctrl-C handling
        return 0;
    }

    // serve command logic
    if (*serve) {
        std::cout << "[Serve] Serving model at: " << serve_model_path << std::endl;
        // TODO: implement serve logic
        return 0;
    }

    if (argc == 1) {
        std::cout << app.help() << std::endl;
    }

    return 0;
}
