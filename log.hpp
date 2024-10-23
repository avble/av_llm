/* Copyright (c) 2024-2024 Harry Le (avble.harry at gmail dot com)

It can be used, modified.
*/

#include "av_connect.hpp"

template <typename... Args>
static void llm_log(const char *format, Args... args)
{
    printf(format, args...);
}

template <typename... Args>
static void LLAMACPP_LOG(const char *format, Args... args)
{
    llm_log(format, args...);
}

template <typename... Args>
static void llamacpp_log(log_level level, const char *format, Args... args)
{
    static log_level log_level_ = LOG_TRACE;
    if (level >= log_level_)
        llm_log(format, args...);
}

template <typename... Args>
static void LLAMACPP_LOG_TRACE(const char *format, Args... args)
{
    llamacpp_log(LOG_TRACE, format, args...);
}

template <typename... Args>
static void LLAMACPP_LOG_DEBUG(const char *format, Args... args)
{
    llamacpp_log(LOG_DEBUG, format, args...);
}

template <typename... Args>
static void LLAMACPP_LOG_INFO(const char *format, Args... args)
{
    llamacpp_log(LOG_INFO, format, args...);
}

template <typename... Args>
static void LLAMACPP_LOG_WARN(const char *format, Args... args)
{
    llamacpp_log(LOG_WARN, format, args...);
}

template <typename... Args>
static void LLAMACPP_LOG_ERROR(const char *format, Args... args)
{
    llamacpp_log(LOG_ERROR, format, args...);
}

class logger_function_trace_llamacpp
{
public:
    logger_function_trace_llamacpp(std::string cls_, std::string func_) : cls(cls_), func(func_)
    {
        LLAMACPP_LOG_TRACE("%s:%s ENTER\n", cls.c_str(), func.c_str());
    }

    ~logger_function_trace_llamacpp() { LLAMACPP_LOG_TRACE("%s:%s LEAVE\n", cls.c_str(), func.c_str()); }

private:
    const std::string cls;
    const std::string func;
};

#define LLAMACPP_LOG_TRACE_FUNCTION logger_function_trace_llamacpp x_trace_123_("", __FUNCTION__);
#define LLAMACPP_TRACE_CLS_FUNC_TRACE logger_function_trace_llamacpp x_trace_123_(typeid(this).name(), __FUNCTION__);
#define LLAMACPP_LOG_TRACE_SCOPE(xxx) logger_function_trace_llamacpp x_trace_123_("", xxx);