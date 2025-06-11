# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import clip
import numpy as np
import os
import torch

from deepfashion2_dataset import DeepFashion2
from utils import extract_img_feats

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/deepfashion2')
parser.add_argument('--batch_size', default=64)
parser.add_argument('--clip_model', default='ViT-L/14')
args = parser.parse_args()

def main():
    test_dataset = DeepFashion2(path = args.data_path, split='test')
    val_dataset = DeepFashion2(path = args.data_path, split='validation')
    out_path = os.path.join(args.data_path, 'feats')
    os.makedirs(out_path, exist_ok=True)

    clip_model, transform = clip.load(args.clip_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model.to(device)

    test_split_train_images = test_dataset.get_all_train_images()
    test_split_test_images = test_dataset.get_retrieval_pool_images()
    val_split_train_images = val_dataset.get_all_train_images()
    val_split_test_images = val_dataset.get_retrieval_pool_images() 
    with torch.no_grad():
        test_split_train_feats = extract_img_feats(clip_model, transform, test_split_train_images, args.batch_size)
        test_split_test_feats = extract_img_feats(clip_model, transform, test_split_test_images, args.batch_size)
        val_split_train_feats = extract_img_feats(clip_model, transform, val_split_train_images, args.batch_size)
        val_split_test_feats = extract_img_feats(clip_model, transform, val_split_test_images, args.batch_size)

    model_name = args.clip_model.replace('/', '-')
    np.save(open(os.path.join(out_path, 'test_split_train_feats_{}.npy'.format(model_name)), 'wb'), test_split_train_feats.cpu())
    np.save(open(os.path.join(out_path, 'test_split_test_feats_{}.npy'.format(model_name)), 'wb'), test_split_test_feats.cpu())
    np.save(open(os.path.join(out_path, 'val_split_train_feats_{}.npy'.format(model_name)), 'wb'), val_split_train_feats.cpu())
    np.save(open(os.path.join(out_path, 'val_split_test_feats_{}.npy'.format(model_name)), 'wb'), val_split_test_feats.cpu())


if __name__ == "__main__":
    main()