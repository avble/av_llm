/*
 *
 */

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "common.hpp"

#include <string>

/*
 * references
 * https://qwenlm.github.io/blog/qwen3-embedding/
 *
 */
int main(int argc, char * argv[])
{
    auto print_usage = [](int argc, char * argv[]) { printf("Usage: \n"); };
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING, print_usage))
    {
        print_usage(argc, argv);
        return 1;
    }

    params.embedding = true;
    params.n_batch   = (params.n_batch < params.n_ctx) ? params.n_ctx : params.n_batch;
    params.n_ubatch  = params.n_batch;

    // load model from file
    llama_model * model = [&params]() {
        llama_model_params mparams = common_model_params_to_llama(params);
        return llama_load_model_from_file(params.model.path.c_str(), mparams);
    }();
    GGML_ASSERT(nullptr != model && "Can not initialize model");

    // context
    llama_context * ctx = [&params, &model]() {
        llama_context_params cparams = common_context_params_to_llama(params);
        // cparams.print_param();
        return llama_init_from_model(model, cparams);
    }();
    GGML_ASSERT(nullptr != ctx && "Can not initilize the context");

    // tokenize the prompt
    auto tokens = [&model, &data = params.prompt]() -> std::vector<llama_token> {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (nullptr == vocab)
            return {};
        int n_token = -llama_tokenize(vocab, data.data(), data.size(), NULL, 0, true, true);
        std::vector<llama_token> tokens(n_token);
        if (llama_tokenize(vocab, data.data(), data.size(), tokens.data(), tokens.size(), true, true) < 0)
            return {};
        return tokens;
    }();

    GGML_ASSERT(tokens.size() > 0 && "error: tokenize. check the promp option");

    const uint64_t n_batch = params.n_batch;
    llama_batch batch      = llama_batch_init(n_batch, 0, 1);

    for (int pos = 0; pos < tokens.size(); pos++)
        common_batch_add(batch, tokens[pos], pos, { 0 }, true);

    // llama_batch_print((const llama_batch *) &batch);
    llama_batch_print(&batch);

    // GGML_ASSERT(llama_decode(ctx, batch) < 0 && "error: can not decode");
    if (llama_decode(ctx, batch) < 0)
    {
        printf("error: can not decode");
        exit(-1);
    }

    // get sequence embedding
    int n_embd = llama_model_n_embd(model);

    // pooling method
    enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    std::vector<float> embeddings(n_embd);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE)
    {
        embeddings.resize(n_embd * tokens.size());
        for (int i = 0; i < tokens.size(); i++)
        {
            float * embd      = llama_get_embeddings_ith(ctx, i);
            float * const out = embeddings.data() + i * n_embd;
            common_embd_normalize(embd, out, n_embd, params.embd_normalize);
        }
    }
    else
    {
        float * embd      = llama_get_embeddings_seq(ctx, 0);
        float * const out = embeddings.data();
        common_embd_normalize(embd, out, n_embd, params.embd_normalize);
    }

    // print output
    if (pooling_type == LLAMA_POOLING_TYPE_NONE)
    {
        for (int i = 0; i < tokens.size(); i++)
        {
            printf("[Embedding][%3d]: ", i);
            for (int j = 0; j < 3; j++)
                printf("%.6f ", *(embeddings.data() + i * n_embd + j));
            printf("\n");
        }
    }
    else
    {
        for (int i = 0; i < std::min(3, n_embd); i++)
            printf("%.6f ", *(embeddings.data() + i));
        printf("...");

        for (int i = 0; i < n_embd && i < 2; i++)
            printf("%.6f ", *(embeddings.data() + n_embd - 1 - i));

        printf("\n");
    }
}

/*
curl 127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" -d '{"input": "Your text string goes here",
"model":"text-embedding-3-small" }'
*/
