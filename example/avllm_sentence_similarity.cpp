/*
 * given: query text & list of passages
 * produce the similar passages for the query
 */

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "common.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

std::vector<std::string> sentences = { "That is a happy dog", "That is a very happy person", "Today is a sunny day" };

struct sentence_chunk
{
    sentence_chunk(std::string _sentence) { sentence = std::move(_sentence); }

    std::string sentence;
    std::vector<float> embedding; // embedding vector
};

int main(int argc, char * argv[])
{
    auto print_usage = [](int argc, char * argv[]) {
        printf("basic Usage: \n");
        printf("	%s [options] \n", argv[0]);
        printf("Options:\n");
        printf("	-m, --model <path>       Path to the model file (required)\n");
        printf("	-p, --prompt <text>      Text to embed (required)\n");
        printf("	--embd-normalize         .i.e --embd-normalize 2 . More detail, see description above\n");
        printf("	--pooling <type>         .i.e --pooling = mean. More detail, see description above\n");
        printf("	-h, --help               Show this help message\n");
        printf("\n");
    };
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING, print_usage))
    {
        print_usage(argc, argv);
        return 1;
    }

    params.embedding    = true;
    params.n_batch      = (params.n_batch < params.n_ctx) ? params.n_ctx : params.n_batch;
    params.n_ubatch     = params.n_batch;
    params.pooling_type = params.pooling_type == LLAMA_POOLING_TYPE_NONE ? LLAMA_POOLING_TYPE_MEAN : params.pooling_type;

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

    const uint64_t n_batch = params.n_batch;
    auto embd_print        = [](std::vector<float> & embeddings, int max_print = 10) {
        int n_embd = embeddings.size();

        for (int i = 0; i < std::min(max_print, n_embd); i++)
            printf("% .6f ", *(embeddings.data() + i));

        printf("... ");

        for (int i = 0; i < n_embd && i < max_print; i++)
            printf("% .6f ", *(embeddings.data() + n_embd - 1 - i));

        printf("\n");
    };

    auto context_tokenize = [&ctx](const std::string & text) -> std::vector<llama_token> {
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (nullptr == vocab)
            return {};

        int n_token = -llama_tokenize(vocab, text.data(), text.size(), NULL, 0, true, true);
        std::vector<llama_token> tokens(n_token);
        if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true) < 0)
            return {};

        return tokens;
    };

    auto context_batch_to_embedding = [&ctx, &params, &embd_print](llama_batch & batch) -> std::vector<float> {
        const llama_model * model = llama_get_model(ctx);

        llama_memory_clear(llama_get_memory(ctx), false);
#ifndef NDEBUG
        llama_batch_print(&batch);
#endif // NDEBUG

        // GGML_ASSERT(llama_decode(ctx, batch) < 0 && "error: can not decode");
        if (llama_decode(ctx, batch) < 0)
        {
            printf("error: can not decode");
            exit(-1);
        }

        // get sequence embedding
        int n_embd = llama_model_n_embd(model);

        std::vector<float> embeddings(n_embd);
        {
            float * embd      = llama_get_embeddings_seq(ctx, 0);
            float * const out = embeddings.data();
            common_embd_normalize(embd, out, n_embd, params.embd_normalize);
        }

        return embeddings;
    };

    std::vector<sentence_chunk> sentence_chunks;

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (auto & sentence : sentences)
    {
        auto tokens = context_tokenize(sentence);
        // token to string
        const llama_vocab * vocab = llama_model_get_vocab(model);
#if 0
        for (auto & token : tokens)
        {
            char buf[120];
            int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                printf("error: failed to convert token to piece\n");
                return {};
            }
            std::string s(buf, n);
            printf("token: %s\n", s.c_str());
        }

#endif
        common_batch_clear(batch);
        for (int pos = 0; pos < tokens.size(); pos++)
            common_batch_add(batch, tokens[pos], pos, { 0 }, true);

        sentence_chunks.emplace_back(sentence);
        sentence_chunks.back().embedding = context_batch_to_embedding(batch);

        // printf("embedding \n");
        // embd_print(sentence_chunks.back().embedding, 3);
        // printf("\n");
    }

    while (true)
    {
        std::string query;
        std::cout << "query> ";
        std::getline(std::cin, query);
        if (query.empty())
        {
            printf("Exiting...\n");
            break;
        }

        auto query_tokens = context_tokenize(query);
        if (query_tokens.empty())
        {
            printf("Failed to tokenize the query\n");
            continue;
        }

        common_batch_clear(batch);
        for (int pos = 0; pos < query_tokens.size(); pos++)
            common_batch_add(batch, query_tokens[pos], pos, { 0 }, true);

        std::vector<float> query_embedding = context_batch_to_embedding(batch);

        // Calculate similarity with each sentence chunk

        std::vector<std::pair<int, float>> similarities;

        for (int i = 0; i < sentence_chunks.size(); i++)
        {
            const auto & chunk = sentence_chunks[i];
            float similarity   = common_embd_similarity_cos(query_embedding.data(), chunk.embedding.data(), chunk.embedding.size());
            similarities.emplace_back(i, similarity);
            // printf("Query: '%s' | Sentence: '%s' | Similarity: %.4f\n", query.c_str(), chunk.sentence.c_str(), similarity);
        }

        std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float> & a, const std::pair<int, float> & b) {
            return a.second > b.second; // Sort by similarity in descending order
        });

        printf("Top similar sentences:\n");
        for (int i = 0; i < std::min(3, (int) similarities.size()); i++)
        {
            printf("  %.4f: '%s' \n", similarities[i].second, sentence_chunks[similarities[i].first].sentence.c_str());
        }
    };

    llama_batch_free(batch);
}

/*
 * model:
 * Qwen2.5 embedding
 * Qwen 3 embedding
 *
 */
