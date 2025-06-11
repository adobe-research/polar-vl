# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import clip
import math
import torch
import torch.nn as nn
from torch.nn.functional import multi_head_attention_forward

class Model(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dim = clip_model.token_embedding.weight.shape[1]
        
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text_tokens):
        return self.clip_model.encode_text(text_tokens)

    def encode_text_with_v_update(self, text_tokens, v_lora_A, v_lora_B):
        # encode text with LoRA update on final layer attention value transform
        last_layer = self.clip_model.transformer.resblocks[-1]
        x = self.encode_text_up_to_last_layer(text_tokens)
        ln1_x = last_layer.ln_1(x)
        v_update = (v_lora_B @ v_lora_A).type(x.dtype)
        qkv_update = torch.cat([torch.zeros_like(v_update), torch.zeros_like(v_update), v_update])

        attn = nn.functional.multi_head_attention_forward(
                ln1_x, ln1_x, ln1_x, last_layer.attn.embed_dim, last_layer.attn.num_heads,
                last_layer.attn.in_proj_weight + qkv_update,
                last_layer.attn.in_proj_bias,
                last_layer.attn.bias_k, last_layer.attn.bias_v, last_layer.attn.add_zero_attn,
                last_layer.attn.dropout, 
                last_layer.attn.out_proj.weight,
                last_layer.attn.out_proj.bias,
                training=last_layer.attn.training,
                need_weights=False,
                attn_mask=last_layer.attn_mask.to(dtype=ln1_x.dtype, device=ln1_x.device)
            )
        
        # finish encoding process
        x = x + attn[0]
        x = x + last_layer.mlp(last_layer.ln_2(x))
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        x = x @ self.clip_model.text_projection

        return x

    def finish_encoding_with_v_update(self, text_tokens, last_layer_x, ln1_x, attn_weights, v_lora_A, v_lora_B):
        # compute last layer with v update
        last_layer = self.clip_model.transformer.resblocks[-1]
        values = torch.matmul(ln1_x, last_layer.attn.in_proj_weight[self.dim*2:].T)
        lora_update = v_lora_B @ v_lora_A
        values2 = values + torch.matmul(ln1_x, lora_update.type(self.clip_model.dtype).T)
        weighted_values = torch.matmul(attn_weights, values2.permute(1, 0, 2)).permute(1, 0, 2)
        attn_out = last_layer.attn.out_proj(weighted_values)

        # finish encoding process
        x = last_layer_x + attn_out
        x = x + last_layer.mlp(last_layer.ln_2(x))
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        x = x @ self.clip_model.text_projection

        return x

    def encode_text_up_to_last_layer(self, text_tokens):
        x = self.clip_model.token_embedding(text_tokens).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2).type(self.clip_model.dtype)
        for i in range(11):
            x = self.clip_model.transformer.resblocks[i](x)
        return x

    def precompute_last_layer_inputs_attn(self, text_tokens):
        # precompute representation up to final layer, input to final attention layer, and attention weights
        x = self.encode_text_up_to_last_layer(text_tokens)
        last_layer = self.clip_model.transformer.resblocks[-1]
        ln1_x = last_layer.ln_1(x)
        last_layer.attn_mask = last_layer.attn_mask.to(dtype=ln1_x.dtype, device=ln1_x.device) if last_layer.attn_mask is not None else None     
        attn_out, attn_weights = last_layer.attn(ln1_x, ln1_x, ln1_x, need_weights=True, attn_mask=last_layer.attn_mask)
        return x, ln1_x, attn_weights


def personalize_for_concept(model, train_prompts, train_img_embeddings, lr, reg_weight, num_iters):
    device = model.clip_model.positional_embedding.device
    v_lora_B = nn.Parameter(torch.empty(model.dim, 1, requires_grad=True, device=device))
    v_lora_A = nn.Parameter(torch.empty(1, model.dim, requires_grad=True, device=device))
    nn.init.zeros_(v_lora_B)
    nn.init.kaiming_uniform_(v_lora_A, a=math.sqrt(5))
    optimizer = torch.optim.Adam([v_lora_A, v_lora_B], lr=lr)

    text_tokens = clip.tokenize(train_prompts).to(device)
    with torch.no_grad():
        x, ln1_x, attn_weights = model.precompute_last_layer_inputs_attn(text_tokens)
    
    for iter in range(num_iters):
        optimizer.zero_grad()
        norm_v_lora_A = nn.functional.normalize(v_lora_A, dim=-1) # norm-1 constraint
        y = model.finish_encoding_with_v_update(text_tokens, x, ln1_x, attn_weights, norm_v_lora_A, v_lora_B)
        pers_loss = ((nn.functional.normalize(y, dim=-1) - train_img_embeddings)**2).mean()
        reg_loss = torch.mean(v_lora_B**2)
        loss = pers_loss + reg_weight * reg_loss
        loss.backward()
        optimizer.step()

    v_lora_A = nn.functional.normalize(v_lora_A, dim=-1)
    return v_lora_A.detach(), v_lora_B.detach()
    
