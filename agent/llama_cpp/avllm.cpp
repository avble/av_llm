#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama-cpp.h"
#include "llama.h"
#include "log.h"
#include "log.hpp"
#include "sampling.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <cstdarg>
#include <inttypes.h>

#ifdef _MSC_VER
#include <ciso646>
#endif

struct xoptions
{
    int n_predict = 1024;
    bool jinja    = false; // jinja template

    // sampling
    double repeat_penalty = 1.0;
    int top_k             = 20;
    float temperature     = 1.0;
    float top_p           = 1.8;
    float min_p           = 0.05;

    // decoding
    int n_ctx       = 10096;
    int n_batch     = 10096;
    int n_ubatch    = 4096;
    int ngl         = 99;
    bool flash_attn = true;
    // server
    int port = 8080;
    // others
    std::string model_url_or_alias;
    std::string model_path_emb;
    // llama-server
    std::string llama_srv_args;
} xoptions_;

struct model_general_t
{

    void init(std::string _model_path)
    {

        model_path = _model_path;

        {
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers       = xoptions_.ngl;
            model_ptr                       = llama_model_ptr(llama_model_load_from_file(model_path.c_str(), model_params));
        }
        if (!model_ptr)
            return;

        {
            llama_model * model = model_ptr.get();
            chat_templates_ptr  = common_chat_templates_init(model, "");
            try
            {
                common_chat_format_example(chat_templates_ptr.get(), false, {});
            } catch (const std::exception & e)
            {
                AVLLM_LOG_WARN("%s: Chat template parsing error: %s\n", __func__, e.what());
                AVLLM_LOG_WARN("%s: The chat template that comes with this model is "
                               "not yet supported, falling back to chatml. "
                               "This may cause the "
                               "model to output suboptimal responses\n",
                               __func__);
                chat_templates_ptr = common_chat_templates_init(model, "chatml");
            }
            {
                AVLLM_LOG_DEBUG("%s: chat template: %s \n", __func__, common_chat_templates_source(chat_templates_ptr.get()));
                AVLLM_LOG_DEBUG("%s: chat example: %s \n", __func__,
                                common_chat_format_example(chat_templates_ptr.get(), false, {}).c_str());
            }
        }

        // initialize the sampler
        {
            auto sparams         = llama_sampler_chain_default_params();
            sparams.no_perf      = false;
            llama_sampler * smpl = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(xoptions_.top_k));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(xoptions_.temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(xoptions_.top_p, xoptions_.min_p));
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
            sampler_default_ptr = llama_sampler_ptr(smpl);
        }

        if (!sampler_default_ptr)
        {
            AVLLM_LOG_ERROR("%s: error: could not create sampling\n", __func__);
            return;
        }

        {
            llama_context_params ctx_params = llama_context_default_params();
            llama_model * model             = model_ptr.get();
            ctx_params.no_perf              = false;
            ctx_params.n_ctx                = xoptions_.n_ctx;
            ctx_params.n_batch              = xoptions_.n_batch;
            ctx_params.n_ubatch             = xoptions_.n_ubatch;
            ctx_params.flash_attn           = true;

            ctx_ptr = llama_context_ptr(llama_init_from_model(model_ptr.get(), ctx_params));
        }

        initialized = true;
    }

    const llama_model * get_model() const
    {
        if (!model_ptr)
        {
            AVLLM_LOG_ERROR("%s: error: model is not initialized\n", __func__);
            return nullptr;
        }
        return model_ptr.get();
    }

    llama_context * get_context()
    {
        if (!ctx_ptr)
        {
            AVLLM_LOG_ERROR("%s: error: context is not initialized\n", __func__);
            return nullptr;
        }
        return ctx_ptr.get();
    }

    llama_sampler * get_sampler()
    {
        if (!sampler_default_ptr)
        {
            AVLLM_LOG_ERROR("%s: error: sampler is not initialized\n", __func__);
            return nullptr;
        }
        return sampler_default_ptr.get();
    }

    std::vector<llama_token> model_string_to_tokens(const std::string & str)
    {
        llama_model * model = model_ptr.get();
        auto tokens         = [&model, &str]() -> std::vector<llama_token> {
            const llama_vocab * vocab = llama_model_get_vocab(model);
            int n_token               = -llama_tokenize(vocab, str.data(), str.size(), NULL, 0, true, true);
            std::vector<llama_token> tokens(n_token);
            if (llama_tokenize(vocab, str.data(), str.size(), tokens.data(), tokens.size(), true, true) < 0)
                return {};
            return tokens;
        }();
        return tokens;
    };

    bool is_initialized() const { return initialized; }

    llama_model_ptr model_ptr                    = nullptr;
    common_chat_templates_ptr chat_templates_ptr = nullptr;
    llama_sampler_ptr sampler_default_ptr        = nullptr;
    std::string model_path;
    bool initialized = false;
    llama_context_ptr ctx_ptr;

} model_general;

static void print_usage(int, char ** argv)
{
    printf("\nexample usage:\n");
    printf("%15s -m model.gguf -input prompt\n"
           "                              -input @file\n",
           argv[0]);
    printf("\n");
}

namespace av_llm {
// avoid conflict definition elsewhere
static std::string string_format(const char * fmt, ...)
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
  //

// global option
std::string model_path;
std::string prompt;

extern "C" {
void av_llm_init(const char * model_path_)
{
    AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("%s - path: %s", __FUNCTION__, model_path_).c_str())

    model_general.init(model_path_);
}

void av_llm_set_prompt(std::vector<int32_t> _prompt_tokens)
{
    AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("%s - token-size: %d", __FUNCTION__, _prompt_tokens.size()).c_str())

    // print token
    llama_context * ctx  = model_general.get_context();
    llama_sampler * smpl = model_general.get_sampler();

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (false)
    {
        std::cout << "\ntokens: \n";
        for (auto token : _prompt_tokens)
        {
            char buf[120];
            int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                AVLLM_LOG_ERROR("%s: error: failed to tokenize \n", __func__);
                return;
            }
            std::string s(buf, n);
            std::cout << s;
        }
        std::cout << "\nend" << std::endl;
    }

    if (!model_general.is_initialized())
    {
        AVLLM_LOG_ERROR("%s: error: model is not initialized\n", __func__);
        return;
    }

    if (!ctx)
    {
        AVLLM_LOG_ERROR("%s: error: context is not initialized\n", __func__);
        return;
    }

    {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_get_one(_prompt_tokens.data(), _prompt_tokens.size());
        llama_decode(ctx, batch);
    }
}

int av_llm_get_next_token()
{
    AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("%s", __FUNCTION__).c_str())

    llama_context * ctx  = model_general.get_context();
    llama_sampler * smpl = model_general.get_sampler();

    llama_token new_token;
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    new_token = llama_sampler_sample(smpl, ctx, -1);

    if (llama_vocab_is_eog(vocab, new_token))
        return -1;

    if (false)
    {
        char buf[120];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        // std::cout << "token: " << new_token << " - " << std::string(buf, n) << std::endl;
    }
    {
        llama_batch batch = llama_batch_get_one(&new_token, 1);
        llama_decode(ctx, batch);
    }

    return new_token;
}

void av_llm_debug(std::vector<int32_t> _prompt_tokens)
{
    AVLLM_LOG_TRACE_SCOPE(av_llm::string_format("%s", __FUNCTION__).c_str())
}
}
