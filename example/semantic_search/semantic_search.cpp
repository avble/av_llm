/*
 * description:
 * + create the chunks of sentences from the file
 * + then embed each chunk
 * given: query text & list of passages
 * return the top chunks with high reposibility
 */

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../common.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// console color
// Text color
#define CONSOLE_COLOR_BLACK "\033[30m"
#define CONSOLE_COLOR_RED "\033[31m"
#define CONSOLE_COLOR_GREEN "\033[32m"
#define CONSOLE_COLOR_YELLOW "\033[33m"
#define CONSOLE_COLOR_BLUE "\033[34m"
#define CONSOLE_COLOR_MAGENTA "\033[35m"
#define CONSOLE_COLOR_CYAN "\033[36m"
#define CONSOLE_COLOR_WHITE "\033[37m"

// Bold text
#define CONSOLE_BOLD "\033[1m"

// Reset all attributes
#define CONSOLE_RESET "\033[0m"

const int chunk_size = 512; // size of each chunk in bytes, can be adjusted

struct file_chunk
{
    file_chunk(std::string _sentence, int _pos, std::string_view _filename = "") :
        sentence(std::move(_sentence)), pos(_pos), filename(_filename)
    {}

    std::string sentence;
    int pos;
    std::string filename;         // filename of the chunk, if any
    std::vector<float> embedding; // embedding vector
};

const char * target_file = "CppCoreGuidelines.md";

int main(int argc, char * argv[])
{
    auto print_usage = [](int argc, char * argv[]) {
        printf("basic Usage: \n");
        printf("	%s [options] \n", argv[0]);
        printf("Options:\n");
        printf("	-m, --model <path>       Path to the model file (required)\n");
        printf("  --top-k <n>              Number of top chunks to return. More detail, see decription above\n");
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

    std::vector<file_chunk> file_chunks;

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    // read file and split into sentences
    //
    std::fstream file(target_file);

    if (!file.is_open())
    {
        printf("error: can not open file %s\n", target_file);
        return 1;
    }

    int pos_sentence = 0;
    char buff[chunk_size + 1]; // +1 for null-terminator];
    while (file.read(&buff[0], chunk_size))
    {
        std::string sentence      = std::string(buff, file.gcount());
        auto tokens               = context_tokenize(sentence);
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

        file_chunks.emplace_back(sentence, pos_sentence, target_file);
        pos_sentence += sentence.size();

        file_chunks.back().embedding = context_batch_to_embedding(batch);
        sentence.clear();
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

        for (int i = 0; i < file_chunks.size(); i++)
        {
            const auto & chunk = file_chunks[i];
            float similarity   = common_embd_similarity_cos(query_embedding.data(), chunk.embedding.data(), chunk.embedding.size());
            similarities.emplace_back(i, similarity);
            // printf("Query: '%s' | Sentence: '%s' | Similarity: %.4f\n", query.c_str(), chunk.sentence.c_str(), similarity);
        }

        std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float> & a, const std::pair<int, float> & b) {
            return a.second > b.second; // Sort by similarity in descending order
        });

        printf("Top similar sentences:\n");
        for (int i = 0; i < std::min(params.sampling.top_k, (int) similarities.size()); i++)
        {
            printf("similarity: %s %.4f %s \n", CONSOLE_COLOR_GREEN, similarities[i].second, CONSOLE_RESET);
            printf("-----------------------\n");
            printf("%s", file_chunks[similarities[i].first].sentence.c_str());
            printf("\n-----------------------\n");
            printf("file: %s %s %s \n", CONSOLE_COLOR_RED, file_chunks[similarities[i].first].filename.c_str(), CONSOLE_RESET);
            printf("\n");
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
