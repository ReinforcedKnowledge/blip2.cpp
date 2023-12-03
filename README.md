# blip2.cpp
BLIP2 Inference in plain C/C++ with [GGML](https://github.com/ggerganov/ggml).

# Description
This is a dependency free implementation of BLIP-2 ViT-L OPT2.7B framework suggested by [Salesforce](https://github.com/salesforce/LAVIS/tree/main) in their [paper, BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597).

# To-Do List

1. ViT-L/14 implementation
This will be a full implementation of the [ViT-L/14 from CLIP](Learning Transferable Visual Models From Natural Language Supervision) but the last layer will be removed when used, according to the authors:
> We remove the last layer of the ViT and uses the second last layer’s output features, which leads to slightly better performance.

- [ ] Load an image
- [ ] Implement the ViT-L/14
- [ ] Get the image representation

2. OPT (Transformer-based Language Model) Implementation

- [ ] Implement tokenizer
- [ ] Tokenize text
- [ ] Implement the [OPT 2.7B ](OPT: Open Pre-trained Transformer Language Models) language model

3. Q-Former Implementation

- [ ] Implement the first stage of the Q-Former
- [ ] Implement the second stage of the Q-Former
- [ ] Get output tokens

4. Global logic
- [ ] Implement and load the fully conneted layer between Q-Former and the language model
- [ ] Connect the three parts
