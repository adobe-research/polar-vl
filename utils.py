import clip
import copy
import numpy as np
from PIL import Image
import torch


def extract_img_feats(clip_model, transform, img_paths, batch_size, normalize=True):
    device = next(clip_model.parameters()).device
    imgs = torch.stack([transform(Image.open(img_path)) for img_path in img_paths]).to(device)
    batches = imgs.split(batch_size)
    feats = []
    for batch in batches:
        feats.append(clip_model.encode_image(batch))
    feats = torch.cat(feats)
    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

    
# The following functions are from https://github.com/hsp-iit/concon-chi_benchmark/blob/main/vlpers/utils/evaluation.py

def get_all_ranks(logits_matrix, ground_truth):
    # The full value exceed the highes value possible 
    preds = torch.full_like(ground_truth, -2, dtype=torch.long)
    
    # return the indexes corresponging to the computed scores
    # ranking[0][2] -> 6 than the image number 6 is the third most similar to label 0 
    all_ranks = torch.argsort(logits_matrix, descending=True)
    
    return all_ranks.numpy()

def get_ranks(logits_matrix, ground_truth):
    # The full value exceed the highes value possible 
    preds = torch.full_like(ground_truth, -2, dtype=torch.long)
    
    # return the indexes corresponging to the computed scores
    # ranking[0][2] -> 6 than the image number 6 is the third most similar to label 0 
    all_ranks = torch.argsort(logits_matrix, descending=True)
    # Row wise equality returns a boolean matrix with True where the correct images for
    # each labels are
    ground_truth_mask = (all_ranks.unsqueeze(2) == ground_truth.unsqueeze(1)).sum(dim=-1).bool()
    count = ground_truth_mask.sum(dim=1)
    
    unfolded_preds = torch.where(ground_truth_mask)[1]
    unfolded_preds = unfolded_preds.detach().cpu()
    
    range_tensor = torch.arange(preds.size(1))[None].repeat(preds.size(0), 1)
    mask = range_tensor < count.unsqueeze(1)
    
    preds[mask] = unfolded_preds
    preds = preds + 1
    
    # Finding the gts corresponding to the ranks
    img_ids = all_ranks[torch.arange(all_ranks.shape[0])[..., None], preds - 1]
    img_ids[~mask] = -1

    return preds.numpy(), img_ids.numpy()

def recall_at(logits_matrix, ground_truth, k=1, avg=True):
    ranks, _ = get_ranks(logits_matrix, ground_truth)
    ranks = copy.deepcopy(ranks).astype(np.float32)
    ranks[ranks == -1] = np.nan
    
    res = np.any(ranks <= k, axis=-1)
    
    if avg:
        res = np.mean(res)
    return res

def mRR(logits_matrix, ground_truth, avg=True):
    # Mean Reciprocal Rank
    ranks, _ = get_ranks(logits_matrix, ground_truth)
    ranks = copy.deepcopy(ranks).astype(np.float32)
    ranks[ranks == -1] = np.nan
    
    res = 1 / np.nanmin(ranks, axis=1)
    
    if avg:
        res = np.mean(res)
    return res

def mAP(logits_matrix, ground_truth, avg=True):
    # Mean Average Precisin
    ranks, _ = get_ranks(logits_matrix, ground_truth)
    ranks = copy.deepcopy(ranks).astype(np.float32)
    
    ranks[ranks == -1] = np.nan
    
    res = np.nanmean(np.arange(1, ranks.shape[1] + 1) / ranks, axis=1)
    
    if avg:
        res = np.mean(res)
    return res


def mAP_at(logits_matrix, ground_truth, k=50, avg=True):
    # Mean Average Precisin
    ranks, _ = get_ranks(logits_matrix, ground_truth)
    ranks = copy.deepcopy(ranks).astype(np.float32)
    ranks[ranks == -1] = np.nan
    
    gts_count = (~np.isnan(ranks)).sum(axis=1)
    ranks[ranks > k] = np.nan
    
    precision = np.arange(1, ranks.shape[1] + 1) / ranks
    res = np.nansum(precision, axis=1) / np.clip(gts_count, 0, k)
    
    if avg:
        res = np.mean(res)
    return res