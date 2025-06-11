# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import json
import numpy as np
import os
import pandas as pd

class DeepFashion2(object):
    def __init__(self, path='./data/deepfashion2', split='test'): # or 'validation'
        self.path = path
        self.split = split

        self.train_images = []
        self.train_concepts = []
        self.class_names = []
        self.codes = []

        class_df = pd.read_csv(os.path.join(path, 'train_coarse_grained_names.csv'))

        with open(os.path.join(path, '{}_fsl_train.txt'.format(split)), 'r') as f:
            concept_idx = 0
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    code = line.split('/')[-1][:-1]
                    self.codes.append(int(code))
                    self.class_names.append(class_df[class_df['unique_pair_ids'] == int(code)]['pair_id_categories'].values[0])
                elif line == '':
                    concept_idx += 1
                else:
                    self.train_images.append(os.path.join(self.path, "train", "image", line))
                    self.train_concepts.append(concept_idx)

        self.num_concepts = len(self.class_names)
        
        # per PALAVRA benchmark, shortened captions are used for evaluation
        split_caption_df = pd.read_csv(os.path.join(path, '{}_captions.csv'.format(split)))
        shortened_caption_df = pd.read_csv(os.path.join(path, 'shortened_deepfashion2_captions.csv'))
        
        self.retrieval_images = split_caption_df['image_name'].values
        self.retrieval_captions = []
        self.retrieval_concepts = []

        for img_path in self.retrieval_images:
            caption_match = shortened_caption_df[shortened_caption_df['image_name'] == img_path]
            assert(len(caption_match) == 1)
            self.retrieval_captions.append(caption_match['caption'].iloc[0])
            code = caption_match['pair_id'].iloc[0]
            self.retrieval_concepts.append(np.where(self.codes == code)[0][0])

        self.retrieval_images = [os.path.join(self.path, img_path) for img_path in self.retrieval_images]
        self.retrieval_captions = [caption.replace("person", "person wearing *") for caption in self.retrieval_captions]


    def get_train_images_for_concept(self, concept):
        return [i for i, c in zip(self.train_images, self.train_concepts) if c == concept]

    def get_train_feats_for_concept(self, concept,  model='ViT-L-14'):
        all_train_feats = np.load(open(os.path.join(self.path, 'feats', '{}_split_train_feats_{}.npy'.format(self.split, model)), "rb"))
        return all_train_feats[np.array(self.train_concepts) == concept]

    def get_concept_class(self, concept):
        return self.class_names[concept]

    def get_all_train_images(self):
        return self.train_images

    def get_retrieval_pool_images(self):
        return self.retrieval_images

    def get_retrieval_pool_feats(self,  model='ViT-L-14'):
        return np.load(open(os.path.join(self.path, 'feats', '{}_split_test_feats_{}.npy'.format(self.split, model)), "rb"))

    def get_retrieval_queries(self):
        image_idx_gts = list(range(len(self.retrieval_captions)))
        return self.retrieval_captions, image_idx_gts, self.retrieval_concepts