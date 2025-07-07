#ifndef _AVLLM_UTILS_H_
#define _AVLLM_UTILS_H_

// #include "_deps/llama_cpp-src/src/llama-vocab.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include "log.hpp"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
#define MIMETYPE_JSON "application/json; charset=utf-8"

// xoptions struct moved here for coding convention
struct xoptions
{
    xoptions()
    {
        repeat_penalty = 1.0;
        n_ctx          = 512;
        n_batch        = 1024;
        ngl            = 0;
        port           = 8080;
    }
    // sampling
    double repeat_penalty;
    // decoding
    int n_ctx;
    int n_batch;
    int ngl;
    // server
    int port;
    // others
    std::string model_url_or_alias;
    std::string model_path_emb;
    // llama-server
    std::string llama_srv_args;
};

// av_connect helper
#define HTTP_SEND_RES_AND_RETURN(res, status, message)                                                                             \
    do                                                                                                                             \
    {                                                                                                                              \
        res->result() = status;                                                                                                    \
        res->set_content(message);                                                                                                 \
        res->end();                                                                                                                \
        return;                                                                                                                    \
    } while (0)

#define HTTP_SEND_RES_AND_CONTINUE(res, status, message)                                                                           \
    do                                                                                                                             \
    {                                                                                                                              \
        res->result() = status;                                                                                                    \
        res->set_content(message);                                                                                                 \
        res->end();                                                                                                                \
    } while (0)

// string
namespace av_llm {
// avoid conflict definition elsewhere
std::string string_format(const char * fmt, ...)
{
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}
} // namespace av_llm

// json
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

template <typename T>
static json json_parse(const T & data)
{
    try
    {
        json j = json::parse(data);
        return j;
    } catch (const json::exception &)
    {
        AVLLM_LOG_WARN("Parsing the json wrong");
    }
    return json();
}

// llama
static void llama_sampler_print(const llama_sampler * smpl)
{
    int n_samplers = llama_sampler_chain_n(smpl);
    for (int i = 0; i < n_samplers; i++)
    {
        llama_sampler * smpl_ = llama_sampler_chain_get(smpl, i);
        printf("%s[%d][%s]", i == 0 ? " " : " -> ", i, llama_sampler_name(smpl_));
    }
}

static void llama_token_print(const llama_vocab * vocab, llama_tokens & tokens)
{
    std::cout << "tokens: \n";
    for (auto token : tokens)
    {
        char buf[120];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0)
        {
            printf("fail tokenize \n");
        }
        std::string s(buf, n);
        std::cout << s;
    }
    std::cout << std::endl;
}

static void llama_batch_print(const llama_batch * batch)
{
    printf("%s\n", std::string(20, '-').c_str());
    printf("%30s: %-10d\n", "n_tokens", batch->n_tokens);

    printf("tokens|emb:\n");
    for (int i = 0; i < batch->n_tokens && batch->token; i++)
        printf("%8d,", batch->token[i]);
    for (int i = 0; i < batch->n_tokens && batch->embd; i++)
        printf("%8f,", batch->embd[i]);
    printf("\npos: \n");
    for (int i = 0; i < batch->n_tokens && batch->pos; i++)
        printf("%8d,", batch->pos[i]);
    printf("\nn_seq\n");
    for (int i = 0; i < batch->n_tokens && batch->seq_id[i]; i++)
        printf("%8d,", batch->seq_id[i][0]);

    printf("\n");
};

// libcurl helper
extern std::filesystem::path app_data_path;
// curl helper function
static size_t write_data(void * ptr, size_t size, size_t nmemb, void * stream)
{
    std::ofstream * of = static_cast<std::ofstream *>(stream);
    try
    {
        of->write((const char *) ptr, size * nmemb);
        return size * nmemb;
    } catch (const std::exception & ex)
    {
        return 0;
    }
}

// Progress callback (older interface)
int progress_callback(void * /*clientp*/, curl_off_t dltotal, curl_off_t dlnow, curl_off_t /*ultotal*/, curl_off_t /*ulnow*/)
{
    if (dltotal == 0)
        return 0; // avoid division by zero

    double progress = (double) dlnow / (double) dltotal * 100.0;
    // std::cout << "\rDownload progress: " << progress << "% (" << dlnow << "/" << dltotal << " bytes)" << std::flush;
    std::cout << "\033[2K\r[";
    for (int i = 0; i < 100; i++)
        std::cout << ((i < (int) progress) ? "#" : ".");
    std::cout << "] " << (int) progress << "%" << std::flush;

    return 0; // return non-zero to abort transfer
}

bool downnload_file_and_write_to_file(std::string url, std::filesystem::path out_file)
{
    AVLLM_LOG_DEBUG("%s: with argument: %s \n", "model_pull", url.c_str());
    CURL * curl;
    CURLcode res;
    bool ret = true;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if (curl)
    {
        auto file_path = app_data_path / out_file;
        // open a file
        std::ofstream of(file_path.c_str());

        if (of.is_open())
        {
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *) &of);
            // curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

            // Set progress callback
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, nullptr);

            // follow redirect
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

            res = curl_easy_perform(curl);
            if (res != CURLE_OK)
            {
                AVLLM_LOG_ERROR("curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
                ret = false;
            }

            of.close();
        }

        AVLLM_LOG_DEBUG("%s: %d \n", "[DEBUG]", __LINE__);
        curl_easy_cleanup(curl);
    }
    else
    {
        AVLLM_LOG_ERROR("%s: %d coud not download model: %s \n", "[DEBUG]", __LINE__, url.c_str());
    }
    curl_global_cleanup();

    return ret;
}

struct human_readable
{
    std::uintmax_t size{};

    template <typename Os>
    friend Os & operator<<(Os & os, human_readable hr)
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

#if 0
    // not support the repository level
    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL)
    {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }

#endif
#if 0
		// not support chunk
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
#endif

#if 0
    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL)
    {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }
#endif

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
#if 0
    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());
#endif

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}
// to here
extern "C" int llama_server_main(int argc, char * argv[]);
#endif
