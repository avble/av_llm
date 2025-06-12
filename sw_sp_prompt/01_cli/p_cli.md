+ Want to make a cli program 
+ using CLI, C++ library 
+ the command structure as below
$ av_llm [--option_1 N] [--option_2 STR] [flags] <command> [param] [sub-command] [sub-param]

# global_options
option_1: this is the option 1. type int
option_2: this is option 2. type string

# command
## model
### subcommand
- ls: list all models
- pull: pull a model. the param is a string type
- del: delete a model


## chat
Descption: start an interactive chat. Get input from std::in. The chat is completed untill the user press ctrl-C twice
Param: a string type. name model-path

# serve
Description: 
param: the string type. name model-path


flags:
    -h,--help: it print help 
    -v,--version: it shows the version

