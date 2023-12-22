#pragma once

#include "ggml/ggml.h"


// BLIP2 layer structures
struct blip2_vision_layer {
    // attention
    struct ggml_tensor* qkv_w;
    struct ggml_tensor* qkv_b;

    struct ggml_tensor* proj_w;
    struct ggml_tensor* proj_b;

    // layernorm 1
    struct ggml_tensor* ln_1_w;
    struct ggml_tensor* ln_1_b;

    // ff
    struct ggml_tensor* ff_1_w;
    struct ggml_tensor* ff_1_b;

    struct ggml_tensor* ff_2_w;
    struct ggml_tensor* ff_2_b;

    // layernorm 2
    struct ggml_tensor* ln_2_w;
    struct ggml_tensor* ln_2_b;
};

// Vision structs
struct blip2_vision_hparams
{
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_layer;
    int32_t n_head;
    int32_t n_intermediate;
    float eps;
};

struct blip2_vison_model {
    struct blip2_vision_hparams hparams;

    // embeddings
    struct ggml_tensor* patch_embeddings_w;
    struct ggml_tensor* patch_embeddings_b;
    struct ggml_tensor* class_embedding;
    struct ggml_tensor* position_embeddings;

    std::vector<blip2_vision_layer> layers;

    struct ggml_tensor* post_ln_w;
    struct ggml_tensor* post_ln_b;
};

// Q-Former structs
struct blip2_qformer_hparams
{
    /* data */
};

struct blip2_qformer_model {

};

// Text structs
typedef int32_t blip2_vocab_id;

struct blip2_tokens {
    blip2_vocab_id* data;
    size_t size;
};

struct blip2_vocab {
    using id = blip2_vocab_id;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;

    //    void add_special_token(const std::string & token);
};

struct blip2_text_hparams
{
    /* data */
};

struct blip2_text_model {

};

// BLIP2 structs
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
    bool vision_gelu = false;
    bool qformer_gelu = false;
    uint32_t num_query_tokens;
    uint32_t cross_attention_frequency;
    struct blip2_vison_model vision_model;
    struct blip2_qformer_model qformer_model;
    struct blip2_text_model text_model;
    struct blip2_vocab vocab;
    float image_mean[3];
    float image_std[3];
    int32_t ftype = 1;
    struct ggml_context* ctx;
    struct gguf_context* ctx_gguf;
    struct blip2_buffer buf_compute;
};


int get_key_idx(const gguf_context * ctx, const char * key);
const uint32_t get_u32(const gguf_context * ctx, std::string key);
const float get_f32(const gguf_context * ctx, std::string key);
struct ggml_tensor* get_tensor(struct ggml_context * ctx, std::string name);
void printShape(struct ggml_tensor *tensor);
void printTensorInfo(struct ggml_tensor* tensor);
void blip2_free(blip2_ctx* ctx);

struct blip2_ctx* blip2_model_load(const char * fname);