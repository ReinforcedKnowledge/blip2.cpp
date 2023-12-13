#pragma once

#include "ggml/ggml.h"

struct blip2_ctx;

struct blip2_vision_hparams
{
    /* data */
};

struct blip2_qformer_hparams
{
    /* data */
};

struct blip2_text_hparams
{
    /* data */
};

typedef int32_t blip2_vocab_id;
struct blip2_tokens {
    blip2_vocab_id* data;
    size_t size;
};

void blip2_model_load(const char * fname);