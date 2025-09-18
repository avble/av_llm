#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <array>
#include <iostream>
#include <vector>

// helper function
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

void common_params_print(const common_params & common)
{

    printf("%30s: %-10d\n", "n_predict", common.n_predict);
    printf("%30s: %-10d\n", "n_ctx", common.n_ctx);
    printf("%30s: %-10d\n", "n_batch", common.n_batch);
    printf("%30s: %-10d\n", "n_ubatch", common.n_ubatch);
    printf("%30s: %-10d\n", "n_keep", common.n_keep);
    printf("%30s: %-10d\n", "n_chunks", common.n_chunks);
    printf("%30s: %-10d\n", "n_parallel", common.n_parallel);
    printf("%30s: %-10d\n", "n_sequences", common.n_sequences);
    printf("%30s: %-10d\n", "grp_attn_n", common.grp_attn_n);
    printf("%30s: %-10d\n", "grp_attn_w", common.grp_attn_w);
    printf("%30s: %-10d\n", "n_print", common.n_print);
    printf("%30s: %-10f\n", "rope_freq_base", common.rope_freq_base);
    printf("%30s: %-10f\n", "rope_freq_scale", common.rope_freq_scale);
    printf("%30s: %-10f\n", "yarn_ext_factor", common.yarn_ext_factor);
    printf("%30s: %-10f\n", "yarn_attn_factor", common.yarn_attn_factor);
    printf("%30s: %-10f\n", "yarn_beta_fast", common.yarn_beta_fast);
    printf("%30s: %-10f\n", "yarn_beta_slow", common.yarn_beta_slow);
    printf("%30s: %-10d\n", "yarn_orig_ctx", common.yarn_orig_ctx);
    printf("%30s: %-10d\n", "n_gpu_layers", common.n_gpu_layers);
}
