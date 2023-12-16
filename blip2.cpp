#include <iostream>
#include <map>

#include "blip2.h"
#include "ggml/ggml.h"


#define KEY_VISION_USE_GELU "blip2.vision.use_gelu"
#define KEY_QFORMER_USE_GELU "blip2.q_former.use_gelu"


void blip2_model_load(const char* fname) {
    struct ggml_context* meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context* ctx = gguf_init_from_file(fname, params);

    if (!ctx) {
        std::cerr << "Failed to initialize gguf_context" << std::endl;
        return;
        }

    const int n_tensors = gguf_get_n_tensors(ctx);
    const int n_kv = gguf_get_n_kv(ctx);

    printf("%s: GGUF version: %d\n", __func__, gguf_get_version(ctx));
    printf("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx));
    printf("%s: n_tensors:    %d\n", __func__, n_tensors);
    printf("%s: n_kv:         %d\n", __func__, n_kv);
    printf("\n");

    // kv
    for (int i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx, i);

        printf("%s: kv[%d]: key = %s\n", __func__, i, key);
    }
    printf("\n");

    // data
    size_t ctx_size = 0;

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx, i);
        printf("Name: %s\n", name);

        const size_t offset = gguf_get_tensor_offset(ctx, i);

        struct ggml_tensor* cur = ggml_get_tensor(meta, name);
        if (cur == NULL) {
            std::cerr << "cur is NULL" << std::endl;
        } else {
            size_t tensor_size = ggml_nbytes(cur);
            std::cout << "Tensor size: " << tensor_size << std::endl;
        }
        ctx_size += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
        size_t padded_size = ggml_nbytes_pad(cur);
        ctx_size += padded_size;

    }
}

int main() {
    const char* filename = "../models/blip2-opt-2.7b_ggml-two_tower_blip2-1.gguf";
    blip2_model_load(filename);

    return 0;
}
