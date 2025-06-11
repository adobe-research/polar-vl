import argparse
import clip
import numpy as np
import os
import torch

from deepfashion2_dataset import DeepFashion2
from utils import recall_at, mRR, mAP
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/deepfashion2')
parser.add_argument('--split', default='test')
parser.add_argument('--checkpoint_dir', default='./checkpoints/train_deepfashion2_ViT-L')
parser.add_argument('--clip_model', default='ViT-L/14')
args = parser.parse_args()

def main():
    dataset = DeepFashion2(path=args.data_path, split=args.split)
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

    retrieval_captions, retrieval_gts, retrieval_concepts = dataset.get_retrieval_queries()
    
    # CONCEPT-ONLY EVALUATION
    gen_tokens = clip.tokenize(["An image of {}".format(concept_identifiers[i]) for i in range(dataset.num_concepts)]).to(device)
    gen_text_embeddings = []
    with torch.no_grad():
        for concept in range(dataset.num_concepts):
            gen_text_embeddings.append(model.encode_text_with_v_update(gen_tokens[concept].unsqueeze(dim=0), v_lora_As[concept], v_lora_Bs[concept])[0])
    gen_text_embeddings = torch.stack(gen_text_embeddings)
    gen_text_embeddings = torch.nn.functional.normalize(gen_text_embeddings, dim=-1)

    all_gen_gts = []
    for concept in range(dataset.num_concepts):
        all_gen_gts.append((np.array(retrieval_concepts) == concept).nonzero()[0].tolist())
    # pad to make gt list same length for all concepts
    max_shape = max([len(ls) for ls in all_gen_gts])
    all_gen_gts_padded = torch.tensor([ls + [-1] * (max_shape - len(ls)) for ls in all_gen_gts])
    sim = gen_text_embeddings.cpu() @ img_embeddings.T
    gen_map = mAP(sim, all_gen_gts_padded)
    gen_mrr = mRR(sim, all_gen_gts_padded)

    print("Concept-only: mAP={}, mRR={}".format(round(gen_map * 100, 2), round(gen_mrr * 100, 2)))

    # CONTEXT EVALUATION
    test_queries, test_gts, test_concepts = dataset.get_retrieval_queries()
    text_embeddings = []
    for i in range(len(test_queries)):
        query, concept = test_queries[i], test_concepts[i]
        query = query.replace('*', concept_identifiers[concept])
        lora_A = v_lora_As[concept]
        lora_B = v_lora_Bs[concept]
        query_tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_embedding = model.encode_text_with_v_update(query_tokens, lora_A, lora_B)
        text_embeddings.append(text_embedding)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    sim = text_embeddings.cpu() @ img_embeddings.T

    all_mrr = mRR(sim, torch.tensor(test_gts).unsqueeze(dim=-1))
    all_r5 = recall_at(sim, torch.tensor(test_gts).unsqueeze(dim=-1), k=5)

    print("Context: mRR={}, recall@5={}".format(round(all_mrr * 100, 2), round(all_r5 * 100, 2)))


if __name__ == "__main__":
    main()