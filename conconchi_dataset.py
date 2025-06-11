import json
import numpy as np
import os
import pandas as pd
import torch

# Some code adapted from https://github.com/hsp-iit/concon-chi_benchmark/blob/main/vlpers/datasets/conconchi.py#L124

class ConConChi(object):
    def __init__(self, path="./data/conconchi"):
        self.path = path
        self.img_prefix = os.path.join(path, 'data', 'images/')
        test_json = json.load(open(os.path.join(path, 'data', 'annotations', 'test.json'), "r"))
        self.test_df = pd.DataFrame(test_json['data']).reset_index(drop=True)
        self.num_concepts = 20

        train_json = json.load(open(os.path.join(path, 'data', 'annotations', 'train.json'), "r"))
        self.train_df = pd.DataFrame(train_json['data']).reset_index(drop=True)
        self.train_df = self.train_df.explode('GTS').reset_index(drop=True)
        self.train_df = self.train_df.explode('CONCEPTS').reset_index(drop=True)
        self.train_df['GTS'] = self.img_prefix + self.train_df['GTS']

        # process retrieval queries following https://github.com/hsp-iit/concon-chi_benchmark/blob/708c8ccd22568415cb5a0981790258f331d65f9f/vlpers/datasets/conconchi.py#L92
        self.test_df['GTS'] = self.test_df['GTS'].apply(lambda x: [self.img_prefix + item for item in x])
        self.test_df['ADDITIONAL GTS'] = self.test_df['ADDITIONAL GTS'].apply(lambda x: [self.img_prefix + item for item in x])
        aux = self.test_df.explode('GTS').reset_index(drop=True)
        aux['ID_GTS'] = aux.index
        aux = aux.set_index('GTS')
        self.test_df = self.test_df[self.test_df['KIND'] != 'negative'].reset_index(drop=True)
        self.test_df['ID_GTS'] = (self.test_df['GTS'] + self.test_df['ADDITIONAL GTS']).apply(lambda gts: [aux.loc[gt]['ID_GTS'] for gt in gts if gt in aux.index])
        self.test_df['CONTEXT'] = self.test_df['KIND'].values + "_" + self.test_df['ENV'].values
        
        self.pool_images = aux.index.tolist()
        self.pool_df = aux.reset_index()
        self.pool_df['CONTEXT'] = self.pool_df['KIND'].values + "_" + self.pool_df['ENV'].values
        self.max_gt = self.test_df['ID_GTS'].apply(len).max()

    def get_all_train_images(self):
        return self.train_df['GTS'].tolist()

    def get_train_images_for_concept(self, concept):
        df = self.train_df[self.train_df['CONCEPTS'] == concept]
        return df['GTS'].tolist()

    def get_train_feats_for_concept(self, concept, model="L"):
        indices = self.train_df.index[self.train_df['CONCEPTS'] == concept]
        all_feats = np.load(open(os.path.join(self.path, 'feats', 'train_feats_{}.npy'.format(model)), "rb"))
        return all_feats[indices]

    def get_retrieval_pool_images(self):
        return self.pool_images

    def get_retrieval_pool_feats(self, model='ViT-L-14'):
        return np.load(open(os.path.join(self.path, 'feats', 'test_feats_{}.npy'.format(model)), "rb"))

    def get_test_queries(self, pad_gt=False):
        queries = self.test_df['LABEL'].tolist()
        gts = self.test_df['ID_GTS'].tolist()
        concepts = self.test_df['CONCEPTS'].tolist()

        if pad_gt:
            padded_gt_list = [ls + [-1] * (self.max_gt - len(ls)) for ls in gts]
            gts = torch.tensor(padded_gt_list)

        return queries, gts, concepts

    def get_retrieval_vlm_captions(self, cap_file='./conconchi_llava_captions.json'):
        caps_json = json.load(open(cap_file, 'r'))
        caps_json = {os.path.join(self.path, k): v for k, v in caps_json.items()}
        return [caps_json[img_path] for img_path in self.pool_images]