# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import clip
import numpy as np
import os
import torch

from conconchi_dataset import ConConChi
from utils import recall_at, mRR, mAP
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/conconchi')
parser.add_argument('--checkpoint_dir', default='./checkpoints/conconchi_checkpoints')
parser.add_argument('--clip_model', default='ViT-L/14')
parser.add_argument('--batch_size', default=200)
args = parser.parse_args()

def main():
    dataset = ConConChi(path=args.data_path)
    clip_model, transform = clip.load(args.clip_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    model = Model(clip_model)
    model = model.float()

    img_embeddings = dataset.get_retrieval_pool_feats(model=args.clip_model.replace('/', '-'))
    captions = dataset.get_retrieval_vlm_captions(cap_file='./conconchi_llava_captions.json')
    tokens = clip.tokenize(captions).to(device)
    batches = tokens.split(args.batch_size)

    # Original CLIP (no personal updates)

    text_embeddings = []
    for batch in batches:
        with torch.no_grad():
            text_embeddings.append(model.encode_text(batch))
    text_embeddings = torch.cat(text_embeddings, dim=0)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

    sim = text_embeddings.cpu() @ img_embeddings.T
    r10 = recall_at(sim, torch.arange(len(text_embeddings)).unsqueeze(dim=-1), k=10)
    print("Original CLIP VLM caption r@10={}".format(round(r10 * 100, 2)))

    # Evaluate each personal update

    all_r10s = []
    for concept in range(dataset.num_concepts):
        ckpt = torch.load(os.path.join(args.checkpoint_dir, 'concept{}.pt'.format(concept)), weights_only=True)
        v_lora_A = ckpt['v_lora_A'].to(device)
        v_lora_B = ckpt['v_lora_B'].to(device)

        text_embeddings = []
        for batch in batches:
            with torch.no_grad():
                text_embeddings.append(model.encode_text_with_v_update(batch, v_lora_A, v_lora_B))
        text_embeddings = torch.cat(text_embeddings, dim=0)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

        sim = text_embeddings.cpu() @ img_embeddings.T
        r10 = recall_at(sim, torch.arange(len(text_embeddings)).unsqueeze(dim=-1), k=10)
        all_r10s.append(r10)

    print("VLM caption r@10={}".format(round(np.mean(all_r10s) * 100, 2)))


if __name__ == "__main__":
    main()