#ifndef _AVLLM_UTILS_H_
#define _AVLLM_UTILS_H_

#include "common.h"
#include "llama.h"
#include "log.h"

#include "log.hpp"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using json = nlohmann::ordered_json;
#define MIMETYPE_JSON "application/json; charset=utf-8"

// llama's helper function

// sampling

void llama_sampler_print(const llama_sampler * smpl)
{
    int n_samplers = llama_sampler_chain_n(smpl);
    for (int i = 0; i < n_samplers; i++)
    {
        llama_sampler * smpl_ = llama_sampler_chain_get(smpl, i);
        printf("%s[%d][%s]", i == 0 ? " " : " -> ", i, llama_sampler_name(smpl_));
    }
}

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

// From llama.cpp
// from here
static bool json_is_array_of_numbers(const json & data)
{
    if (data.is_array())
    {
        for (const auto & e : data)
        {
            if (!e.is_number_integer())
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

// is array having BOTH numbers & strings?
static bool json_is_array_of_mixed_numbers_strings(const json & data)
{
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array())
    {
        for (const auto & e : data)
        {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string)
            {
                return true;
            }
        }
    }
    return false;
}

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
static llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special)
{
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array())
    {
        bool first = true;
        for (const auto & p : json_prompt)
        {
            if (p.is_string())
            {
                auto s = p.template get<std::string>();

                llama_tokens p;
                if (first)
                {
                    p     = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                }
                else
                {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            }
            else
            {
                if (first)
                {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    }
    else
    {
        auto s        = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56]]
 */
static std::vector<llama_tokens> tokenize_input_prompts(const llama_vocab * vocab, const json & json_prompt, bool add_special,
                                                        bool parse_special)
{
    std::vector<llama_tokens> result;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt))
    {
        // string or mixed
        result.push_back(tokenize_mixed(vocab, json_prompt, add_special, parse_special));
    }
    else if (json_is_array_of_numbers(json_prompt))
    {
        // array of tokens
        result.push_back(json_prompt.get<llama_tokens>());
    }
    else if (json_prompt.is_array())
    {
        // array of prompts
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt)
        {
            if (p.is_string() || json_is_array_of_mixed_numbers_strings(p))
            {
                result.push_back(tokenize_mixed(vocab, p, add_special, parse_special));
            }
            else if (json_is_array_of_numbers(p))
            {
                // array of tokens
                result.push_back(p.get<llama_tokens>());
            }
            else
            {
                throw std::runtime_error(
                    "element of \"prompt\" must be a string, an list of tokens, or a list of mixed strings & tokens");
            }
        }
    }
    else
    {
        throw std::runtime_error(
            "\"prompt\" must be a string, an list of tokens, a list of mixed strings & tokens, or a list of prompts");
    }
    if (result.empty())
    {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value)
{
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null())
    {
        try
        {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &)
        {
            AVLLM_LOG_WARN("Wrong type supplied for parameter '%s'. Expected '%s', using default value\n", key.c_str(),
                           json(default_value).type_name());
            return default_value;
        }
    }
    else
    {
        return default_value;
    }
}

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
// static llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
//     // If `add_bos` is true, we only add BOS, when json_prompt is a string,
//     // or the first element of the json_prompt array is a string.
//     llama_tokens prompt_tokens;

//     if (json_prompt.is_array()) {
//         bool first = true;
//         for (const auto & p : json_prompt) {
//             if (p.is_string()) {
//                 auto s = p.template get<std::string>();

//                 llama_tokens p;
//                 if (first) {
//                     p = common_tokenize(vocab, s, add_special, parse_special);
//                     first = false;
//                 } else {
//                     p = common_tokenize(vocab, s, false, parse_special);
//                 }

//                 prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
//             } else {
//                 if (first) {
//                     first = false;
//                 }

//                 prompt_tokens.push_back(p.template get<llama_token>());
//             }
//         }
//     } else {
//         auto s = json_prompt.template get<std::string>();
//         prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
//     }

//     return prompt_tokens;
// }

// borrow this from llama.cpp prorject
static llama_tokens format_infill(const llama_vocab * vocab, const json & input_prefix, const json & input_suffix,
                                  const json & input_extra, const int n_batch, const int n_predict, const int n_ctx,
                                  const bool spm_infill, const llama_tokens & tokens_prompt)
{
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL)
    {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto & chunk : input_extra)
    {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text", std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL)
        {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        }
        else
        {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[]  = { 0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70,
                                                        0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00 };
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL)
    {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(), 3 * (n_batch / 4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch / 4) - (2 + tokens_prompt.size())));

    AVLLM_LOG_INFO("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take,
                   (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch) -2 * n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(), tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab))
    {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    AVLLM_LOG_DEBUG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int) extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}
// to here

#endif
