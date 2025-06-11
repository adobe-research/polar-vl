import argparse
import clip
import numpy as np
import os
import torch

from conconchi_dataset import ConConChi
from utils import extract_img_feats

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/conconchi')
parser.add_argument('--batch_size', default=64)
parser.add_argument('--clip_model', default='ViT-L/14')
args = parser.parse_args()

def main():
    dataset = ConConChi(path = args.data_path)
    out_path = os.path.join(args.data_path, 'feats')
    os.makedirs(out_path, exist_ok=True)

    clip_model, transform = clip.load(args.clip_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model.to(device)

    train_images = dataset.get_all_train_images()
    test_images = dataset.get_retrieval_pool_images()
    with torch.no_grad():
        train_feats = extract_img_feats(clip_model, transform, train_images, args.batch_size)
        test_feats = extract_img_feats(clip_model, transform, test_images, args.batch_size)

    model_name = args.clip_model.replace('/', '-')
    np.save(open(os.path.join(out_path, 'train_feats_{}.npy'.format(model_name)), 'wb'), train_feats.cpu())
    np.save(open(os.path.join(out_path, 'test_feats_{}.npy'.format(model_name)), 'wb'), test_feats.cpu())


if __name__ == "__main__":
    main()
