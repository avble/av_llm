#pragma once

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

namespace avllm {
enum class log_level
{
    TRACE = 0,
    DEBUG,
    INFO,
    WARN,
    ERROR,
};

namespace {
// Default log level based on build type
#ifdef NDEBUG
static log_level current_log_level = log_level::INFO; // Release build
#else
static log_level current_log_level = log_level::DEBUG; // Debug build
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
                current_log_level = log_level::TRACE;
            else if (strcmp(env_level, "DEBUG") == 0)
                current_log_level = log_level::DEBUG;
            else if (strcmp(env_level, "INFO") == 0)
                current_log_level = log_level::INFO;
            else if (strcmp(env_level, "WARN") == 0)
                current_log_level = log_level::WARN;
            else if (strcmp(env_level, "ERROR") == 0)
                current_log_level = log_level::ERROR;
        }
    }
} log_level_initializer;

// Helper function to get log level prefix
const char * get_level_prefix(log_level level)
{
    switch (level)
    {
    case log_level::TRACE:
        return "[TRACE]";
    case log_level::DEBUG:
        return "[DEBUG]";
    case log_level::INFO:
        return "[INFO] ";
    case log_level::WARN:
        return "[WARN] ";
    case log_level::ERROR:
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
        printf("%s [%s] ", get_level_prefix(level), module);
        printf(format, args...);
    }
}

// Legacy support for basic logging
template <typename... Args>
static void avllm_Log(const char * format, Args... args)
{
    log(log_level::INFO, "AVLLM", format, args...);
}

// Function tracing class
class logger_function_trace
{
public:
    logger_function_trace(std::string cls_, std::string func_) : cls(cls_), func(func_)
    {
        log(log_level::TRACE, "AVLLM", "%s:%s ENTER\n", cls.c_str(), func.c_str());
    }

    ~logger_function_trace() { log(log_level::TRACE, "AVLLM", "%s:%s LEAVE\n", cls.c_str(), func.c_str()); }

private:
    const std::string cls;
    const std::string func;
};
} // namespace avllm

// Module-specific log macros (outside namespace to avoid prefix in usage)
#define AVLLM_LOG_TRACE(...) avllm::log(avllm::log_level::TRACE, "AVLLM", __VA_ARGS__)
#define AVLLM_LOG_DEBUG(...) avllm::log(avllm::log_level::DEBUG, "AVLLM", __VA_ARGS__)
#define AVLLM_LOG_INFO(...) avllm::log(avllm::log_level::INFO, "AVLLM", __VA_ARGS__)
#define AVLLM_LOG_WARN(...) avllm::log(avllm::log_level::WARN, "AVLLM", __VA_ARGS__)
#define AVLLM_LOG_ERROR(...) avllm::log(avllm::log_level::ERROR, "AVLLM", __VA_ARGS__)

// Support for legacy AVLLM_LOG macro
#define AVLLM_LOG(...) avllm::avllm_Log(__VA_ARGS__)

// Function tracing macros
#define AVLLM_LOG_TRACE_FUNCTION avllm::logger_function_trace x_trace_123_("", __FUNCTION__);
#define AVLLM_TRACE_CLS_FUNC_TRACE avllm::logger_function_trace x_trace_123_(typeid(this).name(), __FUNCTION__);
#define AVLLM_LOG_TRACE_SCOPE(xxx) avllm::logger_function_trace x_trace_123_("", xxx);
