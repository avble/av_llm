#ifndef _AVLLM_UTILS_H_
#define _AVLLM_UTILS_H_

#include <cstdlib>
#include <filesystem>
#include <iostream>

static std::filesystem::path getHomeDirectory()
{
#ifdef _WIN32
    const char * home = std::getenv("USERPROFILE");
    if (home)
        return std::filesystem::path(home);

    const char * drive = std::getenv("HOMEDRIVE");
    const char * path  = std::getenv("HOMEPATH");
    if (drive && path)
        return std::filesystem::path(std::string(drive) + path);
#else
    const char * home = std::getenv("HOME");
    if (home)
        return std::filesystem::path(home);
#endif
    return {}; // Return empty path if not found
}

// borrow this code from cppreference
struct HumanReadable
{
    std::uintmax_t size{};

    template <typename Os>
    friend Os & operator<<(Os & os, HumanReadable hr)
    {
        int i{};
        double mantissa = hr.size;
        for (; mantissa >= 1024.0; mantissa /= 1024.0, ++i)
        {
        }
        os << std::ceil(mantissa * 10.0) / 10.0 << i["BKMGTPE"];
        return i ? os << "B (" << hr.size << ')' : os;
    }
};

#endif
