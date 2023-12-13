import argparse
import json
import os

import torch
from gguf import *
from transformers import Blip2ForConditionalGeneration, Blip2Processor

VISION = "blip2.vision"
Q_FORMER = "blip2.q_former"
TEXT = "blip2.text"


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


ap = argparse.ArgumentParser(prog="convert_hf_to_gguf.py")
ap.add_argument(
    "-m",
    "--model-dir",
    help="Path to model directory cloned from HF Hub",
    required=True,
)
ap.add_argument(
    "--use-f32", action="store_true", default=False, help="Use f32 instead of f16"
)
ap.add_argument(
    "-o",
    "--output-dir",
    help="Directory to save GGUF files. Default is the original model directory",
    default=None,
)

args = ap.parse_args()

dir_model = args.model_dir


model = Blip2ForConditionalGeneration.from_pretrained(
    dir_model, torch_dtype=torch.float16
)
list_vars = model.state_dict()
processor = Blip2Processor.from_pretrained(dir_model)

with open(dir_model + "/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
    tokens = [key for key in vocab]

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    v_hparams = config["vision_config"]
    q_hparams = config["qformer_config"]
    t_hparams = config["text_config"]


fname_middle = "two_tower_blip2"
ftype_str = "f16"
ftype = 1

output_dir = args.output_dir if args.output_dir is not None else dir_model
os.makedirs(output_dir, exist_ok=True)
output_prefix = os.path.basename(output_dir).replace("ggml_", "")
fname_out = os.path.join(
    output_dir, f"{output_prefix}_ggml-{fname_middle}-{ftype_str[ftype]}.gguf"
)
fout = GGUFWriter(path=fname_out, arch="clip")


# image encoder hparams
fout.add_uint32("blip2.vision.image_size", v_hparams["image_size"])
fout.add_uint32("blip2.vision.patch_size", v_hparams["patch_size"])
fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), v_hparams["hidden_size"])
fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), v_hparams["intermediate_size"])
fout.add_uint32("blip2.vision.projection_dim", v_hparams["projection_dim"])
fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), v_hparams["num_attention_heads"])
fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), v_hparams["layer_norm_eps"])
fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), v_hparams["num_hidden_layers"])

image_mean = processor.image_processor.image_mean
image_std = processor.image_processor.image_std
fout.add_array("blip2.vision.image_mean", image_mean)
fout.add_array("blip2.vision.image_std", image_std)

use_gelu_vision = v_hparams["hidden_act"] == "gelu"
fout.add_bool("blip2.vision.use_gelu", use_gelu_vision)

# q former hparams
fout.add_uint32(
    "blip2.q_former.num_query_tokens",
    q_hparams.get("num_query_tokens", config["num_query_tokens"]),
)
fout.add_uint32(
    "blip2.q_former.cross_attention_frequency", q_hparams["cross_attention_frequency"]
)
fout.add_uint32("blip2.q_former.encoder_hidden_size", q_hparams["encoder_hidden_size"])
fout.add_uint32(k(KEY_EMBEDDING_LENGTH, Q_FORMER), q_hparams["hidden_size"])
fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, Q_FORMER), q_hparams["intermediate_size"])
fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, Q_FORMER), q_hparams["layer_norm_eps"])
fout.add_uint32(k(KEY_CONTEXT_LENGTH, Q_FORMER), q_hparams["max_position_embeddings"])
fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, Q_FORMER), q_hparams["num_attention_heads"])
fout.add_uint32(k(KEY_BLOCK_COUNT, Q_FORMER), q_hparams["num_hidden_layers"])

use_gelu_q_former = q_hparams["hidden_act"] == "gelu"
fout.add_bool("blip2.q_former.use_gelu", use_gelu_q_former)

# text encoder hparams
fout.add_uint32(k(KEY_CONTEXT_LENGTH, TEXT), t_hparams["max_position_embeddings"])
fout.add_uint32(k(KEY_EMBEDDING_LENGTH, TEXT), t_hparams["hidden_size"])
fout.add_uint32("blip2.text.word_embed_proj_dim", t_hparams["word_embed_proj_dim"])
fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, TEXT), t_hparams["num_attention_heads"])
fout.add_uint32(k(KEY_BLOCK_COUNT, TEXT), t_hparams["num_hidden_layers"])
fout.add_token_list(tokens)


for name, data in list_vars.items():
    data = data.squeeze().numpy()
    print(f"{name} - {ftype_str} - shape = {data.shape}")
    fout.add_tensor(name, data)

fout.write_header_to_file()
fout.write_kv_data_to_file()
fout.write_tensors_to_file()
fout.close()

print("Done. Output file: " + fname_out)