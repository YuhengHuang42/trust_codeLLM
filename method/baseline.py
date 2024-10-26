import numpy as np
import torch


def compute_halo_score(encoding, k):
    # HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection
    # https://arxiv.org/pdf/2409.17504
    _, sin_value, V_p = torch.linalg.svd(encoding)
    projection = V_p[:k, :].T.cpu().data.numpy()
    projection = (sin_value[:k] * projection).numpy()
    scores = np.mean(np.matmul(encoding.numpy(), projection), -1, keepdims=True)
    scores = np.sqrt(np.sum(np.square(scores), axis=1))
    return scores