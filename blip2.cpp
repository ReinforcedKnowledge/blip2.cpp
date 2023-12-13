#include <iostream>
#include <map>

#include "blip2.h"
#include "ggml/ggml.h"


#define KEY_VISION_USE_GELU "blip2.vision.use_gelu"
#define KEY_QFORMER_USE_GELU "blip2.q_former.use_gelu"


static std::string format(const char* fmt, ...) {
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
    return std::string(buf.data(), buf.size());
}

int get_key_idx(const gguf_context* ctx, const char* key) {
    int i = gguf_find_key(ctx, key);
    if (i == -1) {
        fprintf(stderr, "key %s not found in file\n", key);
        throw std::runtime_error(format("Missing required key: %s", key));
    }

    return i;
}


struct blip2_layer {
    // attention
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * o_w;
    struct ggml_tensor * o_b;

    // layernorm 1
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_1_b;

    // ff
    struct ggml_tensor * ff_i_w;
    struct ggml_tensor * ff_i_b;

    struct ggml_tensor * ff_o_w;
    struct ggml_tensor * ff_o_b;

    // layernorm 2
    struct ggml_tensor * ln_2_w;
    struct ggml_tensor * ln_2_b;
};

struct blip2_vision_model {
    struct blip2_vision_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings;
    struct ggml_tensor * position_embeddings;

    struct ggml_tensor * pre_ln_w;
    struct ggml_tensor * pre_ln_b;

    std::vector<blip2_layer> layers;

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;
};

struct blip2_qformer_model {
    struct blip2_qformer_hparams hparams;
};

struct blip2_text_model {
    struct blip2_text_hparams hparams;
};

struct blip2_vocab {
    using id = blip2_vocab_id;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;
};

struct blip2_buffer {
    uint8_t * data = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~blip2_buffer() { delete[] data; }
};

struct blip2_ctx {
    struct blip2_vision_model vision_model;
    struct blip2_qformer_model qformer_model;
    struct blip2_text_model text_model;
    struct blip2_vocab vocab;
    float image_mean[3];
    float image_std[3];
    bool vision_use_gelu = false;
    bool qformer_use_gelu = false;
    int32_t ftype = 1;
    struct ggml_context* ctx;
    struct gguf_context* ctx_gguf;
    struct blip2_buffer buf_compute;
};

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
        std::cout << "Nothing happened at the offset" << std::endl;

        struct ggml_tensor* cur = ggml_get_tensor(meta, name);
        std::cout << "Nothing happened when getting the tensor" << std::endl;
        ctx_size += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
        std::cout << "Nothing happened computing the size" << std::endl;
        size_t tensor_size = ggml_nbytes(cur);
        std::cout << "Nothing happened when computing the bytes" << std::endl;
        size_t padded_size = ggml_nbytes_pad(cur);
        std::cout << "Nothing happened when computing the padding bytes" << std::endl;
        ctx_size += padded_size;
        std::cout << "*****" << std::endl;
    }

    blip2_ctx* new_blip2 = new blip2_ctx;

    // model size and capabilities
    int idx = get_key_idx(ctx, KEY_VISION_USE_GELU);
    new_blip2->vision_use_gelu = gguf_get_val_bool(ctx, idx);

    idx = get_key_idx(ctx, KEY_QFORMER_USE_GELU);
    new_blip2->qformer_use_gelu = gguf_get_val_bool(ctx, idx);


    printf("%s: model size:     %.2f MB\n", __func__, (ctx_size / 1024.0 / 1024.0));
    printf("%s: metadata size:  %.2f MB\n", __func__, ggml_get_mem_size(meta) / 1024.0 / 1024.0);
}

int main() {
    const char* filename = "../models/blip2-opt-2.7b_ggml-two_tower_blip2-1.gguf";
    blip2_model_load(filename);

    return 0;
}
