import argparse
import clip
import numpy as np
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import math

from conconchi_dataset import ConConChi
from model import Model, personalize_for_concept
from prompts import PROMPT_LIST

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/conconchi', help='path to dataset')
parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints/train_conconchi_ViT-L', help='where to save the checkpoints')
parser.add_argument('--clip_model', type=str, default='ViT-L/14', help='name of CLIP model to use (e.g. ViT-L/14 or ViT-B/32)')
parser.add_argument('--num_iters', type=int, default=500, help='number of training iterations')
parser.add_argument('--reg_weight', type=float, default=0.35, help='weight of regularization loss')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--concept_identifier', type=str, default='sks', help='V* token identifier to use in text queries for the concept')
args = parser.parse_args()

def main():
    dataset = ConConChi(path=args.data_path)
    clip_model, transform = clip.load(args.clip_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    model = Model(clip_model)

    os.makedirs(args.checkpoint_save_dir, exist_ok=True)
    print("Saving checkpoints to {}".format(args.checkpoint_save_dir))

    for concept in tqdm(range(dataset.num_concepts), desc='personalizing for concepts'):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        train_feats = dataset.get_train_feats_for_concept(concept, model=args.clip_model.replace('/', '-'))
        train_feats = torch.tensor(train_feats).to(device)
        train_prompts = random.sample(PROMPT_LIST, len(train_feats))
        train_prompts = [prompt.replace('*', args.concept_identifier) for prompt in train_prompts]
        v_lora_A, v_lora_B = personalize_for_concept(model, train_prompts, train_feats, args.lr, args.reg_weight, args.num_iters)

        torch.save(
            {'v_lora_A': v_lora_A, 'v_lora_B': v_lora_B, 'concept_identifier': args.concept_identifier},
            os.path.join(args.checkpoint_save_dir, 'concept{}.pt'.format(concept))
        )

if __name__ == "__main__":
    main()