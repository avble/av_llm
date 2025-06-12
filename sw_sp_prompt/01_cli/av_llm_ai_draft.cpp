// ... (Keep all your includes and existing code above this line)
#include <CLI/CLI.hpp>

// ... (Keep all your global variables and static functions above this line)

int main(int argc, char ** argv)
{
    {
#ifdef _WIN32
        home_path = std::getenv("USERPROFILE");
        if (home_path.empty())
            home_path = std::getenv("APPDATA"); // Fallback to APPDATA
#else
        home_path = std::getenv("HOME") ? std::filesystem::path(std::getenv("HOME")) : std::filesystem::path();
#endif
    }

    if (home_path.empty())
    {
        AVLLM_LOG_ERROR("%s: \n", "could not find the homepath");
        exit(1);
    }

    app_path = home_path / ".av_llm";

    pre_config_model_init();

    CLI::App app{"av_llm - CLI program"};

    // Global options
    int option_1 = 0;
    std::string option_2;
    app.add_option("--option_1", option_1, "this is the option 1. type int");
    app.add_option("--option_2", option_2, "this is option 2. type string");

    bool version_flag = false;
    app.add_flag("-v,--version", version_flag, "show version");

    // ---- MODEL command ----
    auto model = app.add_subcommand("model", "Model operations");

    // model ls subcommand
    auto model_ls = model->add_subcommand("ls", "List all models");

    // model pull subcommand
    std::string model_pull_url;
    auto model_pull = model->add_subcommand("pull", "Pull a model (url: string)");
    model_pull->add_option("url", model_pull_url, "Model URL")->required();

    // model del subcommand
    std::string model_del_name;
    auto model_del = model->add_subcommand("del", "Delete a model (model: string)");
    model_del->add_option("model", model_del_name, "Model name")->required();

    // ---- CHAT command ----
    std::string chat_model_path;
    auto chat = app.add_subcommand("chat", "Start an interactive chat. Get input from std::in. The chat is completed until the user presses ctrl-C twice");
    chat->add_option("model-path", chat_model_path, "Model path")->required();

    // ---- SERVE command ----
    std::string serve_model_path;
    auto serve = app.add_subcommand("serve", "Serve model");
    serve->add_option("model-path", serve_model_path, "Model path")->required();

    CLI11_PARSE(app, argc, argv);

    if (version_flag) {
        std::cout << "av_llm version 0.1.0" << std::endl;
        return 0;
    }

    // ---- MODEL logic ----
    if (*model) {
        if (*model_ls) {
            model_cmd_handler("ls");
        } else if (*model_pull) {
            model_cmd_handler("pull " + model_pull_url);
        } else if (*model_del) {
            model_cmd_handler("del " + model_del_name);
        } else {
            std::cerr << "Specify a model subcommand: ls, pull <url>, or del <model>\n";
            model->help();
        }
        return 0;
    }

    // ---- CHAT logic ----
    if (*chat) {
        chat_cmd_handler(chat_model_path);
        return 0;
    }

    // ---- SERVE logic ----
    if (*serve) {
        server_cmd_handler(serve_model_path);
        return 0;
    }

    // If no subcommand, fallback to legacy behavior: treat argv[1] as model description (chat or serve)
    if (argc > 1) {
        // Try alias: if single argument is a .gguf file or model name, run chat
        std::string arg1 = argv[1];
        auto chat_serve_caller = [](std::string model_description, std::function<int(std::string)> chat_or_serve_func) -> int {
            std::filesystem::path model_filename = model_description;
            if (model_filename.extension() == ".gguf") {
                std::optional<std::filesystem::path> model_path = std::filesystem::is_regular_file(model_filename) ? model_filename
                    : std::filesystem::is_regular_file(app_path / model_filename)
                    ? std::optional<std::filesystem::path>(app_path / model_filename)
                    : std::nullopt;

                if (model_path.has_value()){
                    chat_or_serve_func(model_path.value().generic_string());
                }
                return 0;
            }
            // handle preconfig model
            if (pre_config_model.find(model_description) != pre_config_model.end())
            {
                std::string url = pre_config_model[model_description];
                auto last_slash = url.find_last_of("/");
                if (last_slash != std::string::npos)
                {
                    std::filesystem::path model_filename = url.substr(last_slash + 1);

                    std::optional<std::filesystem::path> model_path = std::filesystem::is_regular_file(app_path / model_filename)
                        ? std::optional<std::filesystem::path>(app_path / model_filename)
                        : std::nullopt;

                    if (not model_path.has_value())
                    {
                        model_pull(url);
                    }

                    chat_or_serve_func((app_path / model_filename).generic_string());
                }
                return 0;
            }
            return 0;
        };
        chat_serve_caller(arg1, chat_cmd_handler);
        return 0;
    }

    std::cout << app.help() << std::endl;
    return 0;
}