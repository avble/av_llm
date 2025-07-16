#ifndef _AVLLM_LOG_H_
#define _AVLLM_LOG_H_
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

namespace avllm {
enum class log_level
{
    LOG_TRACE = 0,
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERR,
};

namespace {
// Default log level based on build type
#ifdef NDEBUG
static log_level current_log_level = log_level::LOG_INFO; // Release build
#else
static log_level current_log_level = log_level::LOG_TRACE; // Debug build
#endif

// Initialize log level from environment variable
static struct LogLevelInitializer
{
    LogLevelInitializer()
    {
        const char * env_level = std::getenv("AVLLM_LOG_LEVEL");
        if (env_level)
        {
            if (strcmp(env_level, "TRACE") == 0)
                current_log_level = log_level::LOG_TRACE;
            else if (strcmp(env_level, "DEBUG") == 0)
                current_log_level = log_level::LOG_DEBUG;
            else if (strcmp(env_level, "INFO") == 0)
                current_log_level = log_level::LOG_INFO;
            else if (strcmp(env_level, "WARN") == 0)
                current_log_level = log_level::LOG_WARN;
            else if (strcmp(env_level, "ERROR") == 0)
                current_log_level = log_level::LOG_ERR;
        }
    }
} log_level_initializer;

// Helper function to get log level prefix
const char * get_level_prefix(log_level level)
{
    switch (level)
    {
    case log_level::LOG_TRACE:
        return "[TRACE]";
    case log_level::LOG_DEBUG:
        return "[DEBUG]";
    case log_level::LOG_INFO:
        return "[INFO] ";
    case log_level::LOG_WARN:
        return "[WARN] ";
    case log_level::LOG_ERR:
        return "[ERROR]";
    default:
        return "[?????]";
    }
}
} // namespace

// Base logging function
template <typename... Args>
static void log(log_level level, const char * module, const char * format, Args... args)
{
    if (level >= current_log_level)
    {
        printf("%7s[%-6s] ", get_level_prefix(level), module);
        printf(format, args...);
    }
}

// Legacy support for basic logging
template <typename... Args>
static void avllm_Log(const char * format, Args... args)
{
    log(log_level::LOG_INFO, "AVLLM", format, args...);
}

// Function tracing class
class logger_function_trace
{
public:
    logger_function_trace(std::string cls_, std::string func_) : cls(cls_), func(func_)
    {
        if (!cls.empty() && !func.empty())
        {
            log(log_level::LOG_TRACE, "AVLLM", "%s:%s ENTER\n", cls.c_str(), func.c_str());
        }
        else if (!cls.empty())
        {
            log(log_level::LOG_TRACE, "AVLLM", "%s ENTER\n", cls.c_str());
        }
        else if (!func.empty())
        {
            log(log_level::LOG_TRACE, "AVLLM", "%s ENTER\n", func.c_str());
        }
        else
        {
            log(log_level::LOG_TRACE, "AVLLM", "ENTER\n");
        }
    }

    ~logger_function_trace()
    {
        if (!cls.empty() && !func.empty())
        {
            log(log_level::LOG_TRACE, "AVLLM", "%s:%s LEAVE\n", cls.c_str(), func.c_str());
        }
        else if (!cls.empty())
        {
            log(log_level::LOG_TRACE, "AVLLM", "%s LEAVE\n", cls.c_str());
        }
        else if (!func.empty())
        {
            log(log_level::LOG_TRACE, "AVLLM", "%s LEAVE\n", func.c_str());
        }
        else
        {
            log(log_level::LOG_TRACE, "AVLLM", "LEAVE\n");
        }
    }

private:
    const std::string cls;
    const std::string func;
};
} // namespace avllm

// Module-specific log macros (outside namespace to avoid prefix in usage)
#define AVLLM_LOG_TRACE(...) avllm::log(avllm::log_level::LOG_TRACE, "AVLLM", __VA_ARGS__)
#define AVLLM_LOG_DEBUG(...) avllm::log(avllm::log_level::LOG_DEBUG, "AVLLM", __VA_ARGS__)
#define AVLLM_LOG_INFO(...) avllm::log(avllm::log_level::LOG_INFO, "AVLLM", __VA_ARGS__)
#define AVLLM_LOG_WARN(...) avllm::log(avllm::log_level::LOG_WARN, "AVLLM", __VA_ARGS__)
#define AVLLM_LOG_ERROR(...) avllm::log(avllm::log_level::LOG_ERR, "AVLLM", __VA_ARGS__)

// Support for legacy AVLLM_LOG macro
#define AVLLM_LOG(...) avllm::avllm_Log(__VA_ARGS__)

// Function tracing macros
#define AVLLM_LOG_TRACE_FUNCTION avllm::logger_function_trace x_trace_123_("", __FUNCTION__);
#define AVLLM_TRACE_CLS_FUNC_TRACE avllm::logger_function_trace x_trace_123_(typeid(this).name(), __FUNCTION__);
#define AVLLM_LOG_TRACE_SCOPE(xxx) avllm::logger_function_trace x_trace_123_("", xxx);

#endif
