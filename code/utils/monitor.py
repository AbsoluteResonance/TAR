import os
import torch
from torch.nn import functional as F

import matplotlib.pyplot as plt

def monitor_similarity_distributions(feat, mask, epoch, path_prefix="../misc/similarity_plots/"):
    """
    Computes pairwise similarity distributions and plots histograms.
    
    Args:
        feat: Tensor of shape (B, N, D) - feature embeddings.
        mask: Tensor of shape (B, 1, N) - ground truth labels.
    """
    B, N, _ = feat.shape

    # Create pairwise masks: (B, N, N)
    mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()  # Positive mask
    mem_mask_neg = 1 - mem_mask                              # Negative mask

    # Normalize features and compute pairwise cosine similarity
    feat_norm = F.normalize(feat, dim=-1)
    sim_matrix = torch.matmul(feat_norm, feat_norm.transpose(1, 2))
    
    # Optionally, scale by temperature if desired (e.g., divide by tau)
    tau = 0.6
    sim_matrix = sim_matrix / tau

    # Extract positive and negative similarity scores
    pos_sim = sim_matrix[mem_mask.bool()]
    neg_sim = sim_matrix[mem_mask_neg.bool()]

    # Plot histograms
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(pos_sim.cpu().detach().numpy(), bins=50, alpha=0.7, color='green')
    plt.title('Positive Pair Similarities')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(neg_sim.cpu().detach().numpy(), bins=50, alpha=0.7, color='red')
    plt.title('Negative Pair Similarities')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_prefix, f"epoch_{epoch}_similarity_distributions.png"))
    plt.close()
    # plt.show()