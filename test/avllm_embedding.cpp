#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <iostream>

int main(int argc, char * argv[])
{
    auto print_usage = [](int argc, char * argv[]) { printf("Usage: \n"); };
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, print_usage))
    {
        return 1;
    }
		common_params_print(params);

}
