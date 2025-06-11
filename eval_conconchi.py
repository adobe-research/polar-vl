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
args = parser.parse_args()

def main():
    dataset = ConConChi(path=args.data_path)
    clip_model, transform = clip.load(args.clip_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    model = Model(clip_model)
    model = model.float()

    img_embeddings = dataset.get_retrieval_pool_feats(model=args.clip_model.replace('/', '-'))

    v_lora_As = []
    v_lora_Bs = []
    concept_identifiers = []
    for concept in range(dataset.num_concepts):
        ckpt = torch.load(os.path.join(args.checkpoint_dir, 'concept{}.pt'.format(concept)), weights_only=True)
        v_lora_As.append(ckpt['v_lora_A'].to(device))
        v_lora_Bs.append(ckpt['v_lora_B'].to(device))
        concept_identifiers.append(ckpt['concept_identifier'])
    
    # CONCEPT-ONLY EVALUATION
    gen_tokens = clip.tokenize(["An image of {}".format(concept_identifiers[i]) for i in range(dataset.num_concepts)]).to(device)
    gen_text_embeddings = []
    with torch.no_grad():
        for concept in range(dataset.num_concepts):
            gen_text_embeddings.append(model.encode_text_with_v_update(gen_tokens[concept].unsqueeze(dim=0), v_lora_As[concept], v_lora_Bs[concept])[0])
    gen_text_embeddings = torch.stack(gen_text_embeddings)
    gen_text_embeddings = torch.nn.functional.normalize(gen_text_embeddings, dim=-1)

    # filter retrieval pool to single-concept images only
    single_concept_idxs = dataset.pool_df[dataset.pool_df['CONCEPTS'].apply(lambda x: len(x)==1)].index
    all_gen_gts = []
    for concept in range(dataset.num_concepts):
        full_pool_is_concept = dataset.pool_df['CONCEPTS'].apply(lambda x: concept in x).values
        gen_gts = np.expand_dims(full_pool_is_concept[single_concept_idxs].nonzero()[0], axis=0)
        all_gen_gts.append(full_pool_is_concept[single_concept_idxs].nonzero()[0].tolist())
    # pad to make gt list same length for all concepts
    max_shape = max([len(ls) for ls in all_gen_gts])
    all_gen_gts_padded = torch.tensor([ls + [-1] * (max_shape - len(ls)) for ls in all_gen_gts])
    sim = gen_text_embeddings.cpu() @ img_embeddings[single_concept_idxs].T
    gen_map = mAP(sim, all_gen_gts_padded)
    gen_mrr = mRR(sim, all_gen_gts_padded)

    print("Concept-only: mAP={}, mRR={}".format(round(gen_map * 100, 2), round(gen_mrr * 100, 2)))

    # CONTEXT EVALUATION
    test_queries, test_gts, test_concepts = dataset.get_test_queries(pad_gt=True)
    text_embeddings = []
    for i in range(len(test_queries)):
        query, concepts = test_queries[i], test_concepts[i]
        # replace * with concept identifier(s)
        query = ''.join(part + repl for part, repl in zip(query.split('*'), [concept_identifiers[c] for c in concepts])) + query.split('*')[-1]
        lora_A = torch.cat([v_lora_As[c] for c in concepts], dim=0)
        lora_B = torch.cat([v_lora_Bs[c] for c in concepts], dim=1)
        query_tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_embedding = model.encode_text_with_v_update(query_tokens, lora_A, lora_B)
        text_embeddings.append(text_embedding)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    sim = text_embeddings.cpu() @ img_embeddings.T
    all_map = mAP(sim, test_gts)
    all_mrr = mRR(sim, test_gts)
    all_r1 = recall_at(sim, test_gts)

    print("Context (all): mAP={}, mRR={}, recall@1={}".format(round(all_map * 100, 2), round(all_mrr * 100, 2), round(all_r1 * 100, 2)))

    single_concept_query_idxs = torch.tensor(list(map(lambda x: len(x) == 1, test_concepts)))
    single_map = mAP(sim[single_concept_query_idxs], test_gts[single_concept_query_idxs])
    single_r1 = recall_at(sim[single_concept_query_idxs], test_gts[single_concept_query_idxs])
    single_mrr = mRR(sim[single_concept_query_idxs], test_gts[single_concept_query_idxs])

    print("Context (single-concept): mAP={}, mRR={}, recall@1={}".format(round(single_map * 100, 2), round(single_mrr * 100, 2), round(single_r1 * 100, 2)))

    multi_concept_query_idxs = torch.tensor(list(map(lambda x: len(x) > 1, test_concepts)))
    multi_map = mAP(sim[multi_concept_query_idxs], test_gts[multi_concept_query_idxs])
    multi_r1 = recall_at(sim[multi_concept_query_idxs], test_gts[multi_concept_query_idxs])
    multi_mrr = mRR(sim[multi_concept_query_idxs], test_gts[multi_concept_query_idxs])

    print("Context (multi-concept): mAP={}, mRR={}, recall@1={}".format(round(multi_map * 100, 2), round(multi_mrr * 100, 2), round(multi_r1 * 100, 2)))


if __name__ == "__main__":
    main()