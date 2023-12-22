#include <iostream>
#include <fstream>
#include <map>

#include "blip2.h"
#include "ggml/ggml.h"


// Hparams names
#define KEY_VISION_USE_GELU "blip2.vision.use_gelu"
#define KEY_QFORMER_USE_GELU "blip2.q_former.use_gelu"
#define KEY_IMAGE_SIZE "blip2.vision.image_size"
#define KEY_PATCH_SIZE "blip2.vision.patch_size"
#define CROSS_ATTENTION_FREQUENCY "blip2.q_former.cross_attention_frequency"
#define NUM_QUERY_TOKENS "blip2.q_former.num_query_tokens"
#define KEY_IMAGE_MEAN "blip2.vision.image_mean"
#define KEY_IMAGE_STD "blip2.vision.image_std"
#define KEY_EMBEDDING_LENGTH "blip2.%s.embedding_length"
#define KEY_BLOCK_COUNT "blip2.%s.block_count"
#define KEY_ATTENTION_HEAD_COUNT "blip2.%s.attention.head_count"
#define KEY_FEED_FORWARD_LENGTH "blip2.%s.feed_forward_length"
#define KEY_ATTENTION_LAYERNORM_EPS "blip2.%s.attention.layer_norm_epsilon"

// Tensor names
// Vision
#define V_PATCH_EMBD "vision_model.embeddings.patch_embedding.%s"
#define V_CLASS_EMBD "vision_model.embeddings.class_embedding"
#define V_POS_EMBD "vision_model.embeddings.position_embedding"
#define V_QKV "vision_model.encoder.layers.%d.self_attn.qkv.%s"
#define V_MHA_PROJ "vision_model.encoder.layers.%d.self_attn.projection.%s"
#define V_MHA_LN1 "vision_model.encoder.layers.%d.layer_norm1.%s"
#define V_MHA_FF1 "vision_model.encoder.layers.%d.mlp.fc1.%s"
#define V_MHA_FF2 "vision_model.encoder.layers.%d.mlp.fc2.%s"
#define V_MHA_LN2 "vision_model.encoder.layers.%d.layer_norm2.%s"
#define V_LN_POST "vision_model.post_layernorm.%s"


static std::string format(const char * fmt, ...) {
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

int get_key_idx(const gguf_context * ctx, const char * key) {
    int i = gguf_find_key(ctx, key);
    if (i == -1) {
        fprintf(stderr, "key %s not found in file\n", key);
        throw std::runtime_error(format("Missing required key: %s", key));
    }

    return i;
}

const uint32_t get_u32(const gguf_context * ctx, std::string key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_u32(ctx, i);
}

const float get_f32(const gguf_context * ctx, std::string key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_f32(ctx, i);
}

struct ggml_tensor* get_tensor(struct ggml_context* ctx, std::string name) {
    struct ggml_tensor* cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        printf("unable to find tensor %s\n", name.c_str());
        throw std::runtime_error(format("unable to find tensor %s\n", name.c_str()));
    }

    return cur;
}

// Function to print the shape of a tensor
void printShape(struct ggml_tensor* tensor) {
    int p = GGML_MAX_DIMS - 1; // Start from the last element

    while (p >= 0 && tensor->ne[p] == 1) {
        p--; // Skip elements with a value of 1
    }

    if (p >= 0) {
        std::cout << "(" << tensor->ne[p];
        for (int i = p - 1; i >= 0; i--) {
            std::cout << ", " << tensor->ne[i];
        }
        std::cout << ")" << std::endl;
    }
}

// Function to print tensor information
void printTensorInfo(struct ggml_tensor* tensor) {
    if (tensor) {
        std::cout << "Name: " << tensor->name << std::endl;
        printShape(tensor);
    } else {
        std::cout << "Tensor is nullptr" << std::endl;
    }
}

void blip2_free(blip2_ctx* ctx) {
    ggml_free(ctx->ctx);
    gguf_free(ctx->ctx_gguf);
    delete ctx;
}


struct blip2_ctx* blip2_model_load(const char* fname) {
    struct ggml_context* meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context* ctx = gguf_init_from_file(fname, params);

    if (!ctx) {
        std::cerr << "Failed to initialize gguf_context" << std::endl;
        return nullptr;
        }

    // // kv
    // {
    //     const int n_kv = gguf_get_n_kv(ctx);

    //     for (int i = 0; i < n_kv; ++i) {
    //         const char * key = gguf_get_key(ctx, i);

    //         printf("%s: kv[%d]: key = %s\n", __func__, i, key);
    //     }
    //     printf("\n");
    // }

    // Compute context size
    size_t ctx_size = 0;
    size_t vision_ctx_size = 0;
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);

            struct ggml_tensor * cur = ggml_get_tensor(meta, name);
            ctx_size += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            size_t tensor_size = ggml_nbytes(cur);
            size_t padded_size = ggml_nbytes_pad(cur);
            ctx_size += padded_size;

            // if (strncmp(name, "vision_model", strlen("vision_model")) == 0) {
            //     vision_ctx_size += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            //     vision_ctx_size += padded_size;
            // }

            // printf("Tensor Name: %s\n", name);
        }
    }
    // printf("%s: model size:     %.2f MB\n", __func__, (ctx_size / 1024.0 / 1024.0));
    // printf("%s: vision encoder size:     %.2f MB\n", __func__, (vision_ctx_size / 1024.0 / 1024.0));
    // printf("%s: metadata size:  %.2f MB\n", __func__, ggml_get_mem_size(meta) / 1024.0 / 1024.0);
    
    blip2_ctx* new_blip2 = new blip2_ctx;

    // Model configuration
    {
        int idx = gguf_find_key(ctx, KEY_VISION_USE_GELU);
        new_blip2->vision_gelu = gguf_get_val_bool(ctx, idx);

        idx = gguf_find_key(ctx, KEY_QFORMER_USE_GELU);
        new_blip2->qformer_gelu = gguf_get_val_bool(ctx, idx);

        idx = gguf_find_key(ctx, NUM_QUERY_TOKENS);
        new_blip2->num_query_tokens = gguf_get_val_u32(ctx, idx);

        idx = gguf_find_key(ctx, CROSS_ATTENTION_FREQUENCY);
        new_blip2->cross_attention_frequency = gguf_get_val_u32(ctx, idx);

        // std::cout << "vision_gelu " << std::boolalpha << new_blip2->vision_gelu << std::endl;
        // std::cout << "qformer_gelu " << std::boolalpha << new_blip2->qformer_gelu << std::endl;
        // std::cout << "num_query_tokens " << new_blip2->num_query_tokens << std::endl;
        // std::cout << "cross_attention_frequency " << new_blip2->cross_attention_frequency << std::endl;
    }

    // Load tensors
    {
        struct ggml_init_params params = {
            .mem_size = ctx_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        new_blip2->ctx = ggml_init(params);
        if (!new_blip2->ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            blip2_free(new_blip2);
            return nullptr;
        }

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            printf("cannot open model file for loading tensors\n");
            blip2_free(new_blip2);
            return nullptr;
        }

        const int n_tensors = gguf_get_n_tensors(ctx);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * t = ggml_get_tensor(meta, name);
            struct ggml_tensor * cur = ggml_dup_tensor(new_blip2->ctx, t);
            ggml_set_name(cur, name);
            
            size_t data_offset = gguf_get_data_offset(ctx);
            size_t tensor_offset = gguf_get_tensor_offset(ctx, i);

            // const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
            const size_t offset = data_offset + tensor_offset;
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                printf("%s: failed to seek for tensor %s\n", __func__, name);
                blip2_free(new_blip2);
                return nullptr;
            }

            // Because of limited memory, read each component's tensors for debugging before reading everything
            // fin.read(reinterpret_cast<char *>(cur->data), ggml_nbytes(t));
        }

        fin.close();
    }

    // Load vision model
    {
        auto &vision_model = new_blip2->vision_model;
        auto &hparams = vision_model.hparams;

        hparams.image_size = get_u32(ctx, KEY_IMAGE_SIZE);
        hparams.patch_size = get_u32(ctx, KEY_PATCH_SIZE);
        
        hparams.hidden_size = get_u32(ctx, format(KEY_EMBEDDING_LENGTH, "vision"));
        hparams.n_layer = get_u32(ctx, format(KEY_BLOCK_COUNT, "vision"));
        hparams.n_head = get_u32(ctx, format(KEY_ATTENTION_HEAD_COUNT, "vision"));
        hparams.n_intermediate = get_u32(ctx, format(KEY_FEED_FORWARD_LENGTH, "vision"));
        hparams.eps = get_f32(ctx, format(KEY_ATTENTION_LAYERNORM_EPS, "vision"));

        int idx_mean = get_key_idx(ctx, KEY_IMAGE_MEAN);
        int idx_std = get_key_idx(ctx, KEY_IMAGE_STD);
        for (int i = 0; i < 3; ++i) {
            new_blip2->image_mean[i] = *((float *)gguf_get_arr_data(ctx, idx_mean));
            new_blip2->image_std[i] = *((float *)gguf_get_arr_data(ctx, idx_std));
        }

        // // Print vision hparams
        // {
        //     printf("\n%s: vision model hparams\n", __func__);
        //     printf("image_size         %d\n", hparams.image_size);
        //     printf("patch_size         %d\n", hparams.patch_size);
        //     printf("v_hidden_size      %d\n", hparams.hidden_size);
        //     printf("v_n_intermediate   %d\n", hparams.n_intermediate);
        //     printf("v_n_head           %d\n", hparams.n_head);
        //     printf("v_n_layer          %d\n", hparams.n_layer);
        // }

        // Load vision weights
        vision_model.patch_embeddings_w = get_tensor(new_blip2->ctx, format(V_PATCH_EMBD, "weight"));
        vision_model.patch_embeddings_b = get_tensor(new_blip2->ctx, format(V_PATCH_EMBD, "bias"));
        vision_model.class_embedding = get_tensor(new_blip2->ctx, V_CLASS_EMBD);
        vision_model.position_embeddings = get_tensor(new_blip2->ctx, V_POS_EMBD);

        // Print tensor information for embeddings
        std::cout << "Embeddings Tensor Info: " << std::endl;
        printTensorInfo(vision_model.patch_embeddings_w);
        printTensorInfo(vision_model.patch_embeddings_b);
        printTensorInfo(vision_model.class_embedding);
        printTensorInfo(vision_model.position_embeddings);

        vision_model.layers.resize(hparams.n_layer);
        for (int i = 0; i < hparams.n_layer; ++i) {
            auto & layer = vision_model.layers[i];
            layer.qkv_w = get_tensor(new_blip2->ctx, format(V_QKV, i, "weight"));
            layer.qkv_b = get_tensor(new_blip2->ctx, format(V_QKV, i, "bias"));
            
            layer.proj_w = get_tensor(new_blip2->ctx, format(V_MHA_PROJ, i, "weight"));
            layer.proj_b = get_tensor(new_blip2->ctx, format(V_MHA_PROJ, i, "bias"));

            layer.ln_1_w = get_tensor(new_blip2->ctx, format(V_MHA_LN1, i, "weight"));
            layer.ln_1_b = get_tensor(new_blip2->ctx, format(V_MHA_LN1, i, "bias"));
            
            layer.ff_1_w = get_tensor(new_blip2->ctx, format(V_MHA_FF1, i, "weight"));
            layer.ff_1_b = get_tensor(new_blip2->ctx, format(V_MHA_FF1, i, "bias"));
            layer.ff_2_w = get_tensor(new_blip2->ctx, format(V_MHA_FF2, i, "weight"));
            layer.ff_2_b = get_tensor(new_blip2->ctx, format(V_MHA_FF2, i, "bias"));
            
            layer.ln_2_w = get_tensor(new_blip2->ctx, format(V_MHA_LN2, i, "weight"));
            layer.ln_2_b = get_tensor(new_blip2->ctx, format(V_MHA_LN2, i, "bias"));

            // Print tensor information for each layer
            std::cout << "Layer " << i << " Tensor Info:" << std::endl;
            printTensorInfo(layer.qkv_w);
            printTensorInfo(layer.qkv_b);
            printTensorInfo(layer.proj_w);
            printTensorInfo(layer.proj_b);
            printTensorInfo(layer.ln_1_w);
            printTensorInfo(layer.ln_1_b);
            printTensorInfo(layer.ff_1_w);
            printTensorInfo(layer.ff_1_b);
            printTensorInfo(layer.ff_2_w);
            printTensorInfo(layer.ff_2_b);
            printTensorInfo(layer.ln_2_w);
            printTensorInfo(layer.ln_2_b);
        }

        vision_model.post_ln_w = get_tensor(new_blip2->ctx, format(V_LN_POST, "weight"));
        vision_model.post_ln_b = get_tensor(new_blip2->ctx, format(V_LN_POST, "bias"));

        // Print tensor information for post MHA blocks layer norm
        std::cout << "Post MHA blocks layer norm Tensor Info: " << std::endl;
        printTensorInfo(vision_model.post_ln_w );
        printTensorInfo(vision_model.post_ln_b);
    }

    ggml_free(meta);
    new_blip2->ctx_gguf = ctx;

    return new_blip2;
}

int main() {
    const char* filename = "../models/blip2-opt-2.7b_ggml-two_tower_blip2-1.gguf";
    blip2_ctx* new_blip2  = blip2_model_load(filename);
    blip2_free(new_blip2);
    return 0;
}
