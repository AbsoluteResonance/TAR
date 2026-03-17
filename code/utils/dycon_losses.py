
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_beta(epoch, total_epochs, max_beta=5.0, min_beta=0.5):
    ratio = min_beta / max_beta
    exponent = epoch / total_epochs
    beta = max_beta * (ratio ** exponent)
    return beta
    
def gambling_softmax(logits):
    """
    Compute gambling softmax probabilities over the channel dimension.
    
    Args:
        logits (Tensor): Input tensor of shape (B, C, ...).
    
    Returns:
        Tensor: Softmax probabilities of the same shape.
    """
    exp_logits = torch.exp(logits)
    denom = torch.sum(exp_logits, dim=1, keepdim=True)
    return exp_logits / (denom + 1e-18)

def sigmoid_rampup(current_epoch, total_rampup_epochs, min_threshold, max_threshold, steepness=5.0):
    """
    Compute a dynamic threshold using a sigmoid ramp-up schedule.

    Args:
        current_epoch (int or float): The current training epoch.
        total_rampup_epochs (int or float): The number of epochs over which to ramp up the threshold.
        min_threshold (float): The initial threshold value, chosen based on the histogram's lower tail.
        max_threshold (float): The target threshold value after ramp-up.
        steepness (float, optional): Controls how quickly the threshold ramps up (default=5.0).

    Returns:
        float: The computed threshold for the current epoch.
    """
    if total_rampup_epochs == 0:
        return max_threshold
    current_epoch = max(0.0, min(float(current_epoch), total_rampup_epochs))
    phase = 1.0 - (current_epoch / total_rampup_epochs)
    ramp = math.exp(-steepness * (phase ** 2))
    return min_threshold + (max_threshold - min_threshold) * ramp


class UnCLoss(nn.Module):
    """
    UnCLoss implements an uncertainty-aware consistency loss that compares the prediction distributions 
    from a student and a teacher network. It is designed for semi-supervised learning scenarios, where 
    the teacher network provides guidance (e.g., via noise-added views) to the student network.
    
    The loss is computed as follows:
    
      1. Compute the softmax probability distributions for both student (p_s) and teacher (p_t) logits.
         - s_logits: tensor of shape (B, C, H, W, D) from the student network.
         - t_logits: tensor of shape (B, C, H, W, D) from the teacher network.
      
      2. Compute the Shannon entropy for each distribution:
         - H_s = -∑_c p_s * log(p_s + EPS) for the student (resulting in shape (B, 1, H, W, D)).
         - H_t is computed similarly for the teacher.
         These entropies represent the uncertainty in the respective predictions.
      
      3. Scale and exponentiate the entropies using a parameter β:
         - exp_H_s = exp(β * H_s)
         - exp_H_t = exp(β * H_t)
         Higher values of β increase the impact of the uncertainty in the subsequent weighting.
      
      4. Compute an entropy-weighted squared difference between the student and teacher probability distributions:
         - The basic difference (p_s - p_t)^2 is divided by (exp_H_s + exp_H_t), meaning that regions 
           with lower uncertainty (i.e. lower entropy) receive a higher weight.
      
      5. Add a regularization term proportional to the sum of the entropies, scaled by β:
         - This term penalizes high uncertainty, encouraging the networks to make confident predictions.
      
      6. The final loss is the mean over all elements:
         - First summing the weighted differences over the class dimension, then averaging over spatial dimensions and batch.
    
    Args:
        s_logits (Tensor): Logits from the student network with shape (B, C, H, W, D).
        t_logits (Tensor): Logits from the teacher network with shape (B, C, H, W, D).
        beta (float): A scaling parameter that modulates the influence of the entropy terms. A higher beta
                      increases the weighting effect of the entropy, emphasizing regions of high certainty.
    
    Returns:
        Tensor: A scalar tensor representing the mean uncertainty-aware consistency loss.
    """
    def __init__(self):
        super(UnCLoss, self).__init__()

    def forward(self, s_logits, t_logits, beta):
        EPS = 1e-6

        # Compute student softmax probabilities and their entropy.
        p_s = F.softmax(s_logits, dim=1)  # (B, C, H, W, D)
        p_s_log = torch.log(p_s + EPS)
        H_s = -torch.sum(p_s * p_s_log, dim=1, keepdim=True)  # (B, 1, H, W, D)

        # Compute teacher softmax probabilities and their entropy.
        p_t = F.softmax(t_logits, dim=1)  # (B, C, H, W, D)
        p_t_log = torch.log(p_t + EPS)
        H_t = -torch.sum(p_t * p_t_log, dim=1, keepdim=True)  # (B, 1, H, W, D)

        # Exponentiate the entropies scaled by beta.
        exp_H_s = torch.exp(beta * H_s)
        exp_H_t = torch.exp(beta * H_t)

        # Compute the entropy-weighted squared difference between student and teacher distributions.
        # The higher the certainty (lower entropy), the larger the weight on the difference.
        loss = (p_s - p_t)**2 / (exp_H_s + exp_H_t)

        # Sum the differences over the class dimension, add a penalty for high entropy, and average.
        loss = torch.mean(loss.sum(dim=1) + beta * (H_s + H_t))

        return loss.mean()

class FeCLoss(nn.Module):
    """
    FeCLoss with an auxiliary teacher-based hard negative branch and gambling softmax uncertainty mask for guiding positive samples.
    
    The primary loss is an InfoNCE-style contrastive loss computed from student embeddings.
    The focal weighting is applied to hard positives (same-class pairs with low similarity)
    and hard negatives (different-class pairs with high similarity).
    
    Additionally, an auxiliary loss is computed by comparing student embeddings with teacher embeddings.
    
    A gambling softmax is computed to produce an uncertainty mask from decoded_logits. 
    The entropy (uncertainty) is then used to modulate the loss contribution of positive pairs.
    
    Args:
        device: Computation device ('cpu' or 'cuda').
        temperature: Scaling factor τ.
        gamma: Exponent for focal weighting.
        use_focal: Boolean flag to enable focal weighting on the primary loss.
        rampup_epochs: Number of epochs over which the thresholds are ramped up.
        lambda_cross: Weight for the auxiliary teacher-based negative loss.
    """
    def __init__(self, device, temperature=0.6, gamma=2.0, use_focal=False, rampup_epochs=2000, lambda_cross=1.0):
        super(FeCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.gamma = gamma
        self.use_focal = use_focal
        self.rampup_epochs = rampup_epochs
        self.lambda_cross = lambda_cross

    def forward(self, feat, mask, teacher_feat=None, gambling_uncertainty=None, epoch=0):
        """
        Compute the total loss as the sum of:
         - The FeCLoss computed on student embeddings.
         - An auxiliary cross-negative loss computed between student and teacher embeddings.
         - Modulate the positive part of the student loss using an uncertainty mask
           computed via gambling softmax from student decoder.
        
        Args:
            feat: Tensor of shape (B, N, D) - Student embeddings.
            mask: Tensor of shape (B, 1, N) - Ground truth labels per patch.
            teacher_feat: (Optional) Tensor of shape (B, N, D) - Teacher embeddings.
            epoch: Current epoch for dynamic threshold computation.
            gambling_uncertainty: (Optional) Tensor of shape (B, N) - entropy from student decoder.
        
        Returns:
            Total loss (scalar): student loss + lambda_cross * teacher auxiliary loss,
            with positive samples optionally weighted by the uncertainty mask.
        """
        B, N, _ = feat.shape

        # Primary FeCLoss (Student Only)
        mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()  # (B, N, N): 1 if same label.
        mem_mask_neg = 1 - mem_mask  # (B, N, N): 1 if different labels.

        feat_logits = torch.matmul(feat, feat.transpose(1, 2)) / self.temperature  # (B, N, N)
        identity = torch.eye(N, device=self.device)
        neg_identity = 1 - identity  # Zero out self-similarity.
        feat_logits = feat_logits * neg_identity

        feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
        feat_logits = feat_logits - feat_logits_max.detach()

        exp_logits = torch.exp(feat_logits)  # (B, N, N)
        neg_sum = torch.sum(exp_logits * mem_mask_neg, dim=-1)  # (B, N)

        denominator = exp_logits + neg_sum.unsqueeze(dim=-1)
        division = exp_logits / (denominator + 1e-18)  # Softmax-like probability.

        loss_matrix = -torch.log(division + 1e-18)
        loss_matrix = loss_matrix * mem_mask * neg_identity

        loss_student = torch.sum(loss_matrix, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
        loss_student = loss_student.mean()

        # Apply focal weighting to the student loss
        if self.use_focal:
            similarity = division  # Using normalized similarity as proxy.
            focal_weights = torch.ones_like(similarity)
            pos_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=1.3, max_threshold=1.5)
            neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5)
            hard_pos_mask = mem_mask.bool() & (similarity < pos_thresh)
            focal_weights[hard_pos_mask] = (1 - similarity[hard_pos_mask]).pow(self.gamma)
            hard_neg_mask = mem_mask_neg.bool() & (similarity > neg_thresh)
            focal_weights[hard_neg_mask] = similarity[hard_neg_mask].pow(self.gamma)
            loss_student = torch.sum(loss_matrix * focal_weights, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
            loss_student = loss_student.mean()

        # Incorporate Gambling Softmax Uncertainty Mask for Positives
        if gambling_uncertainty is not None:
            loss_student_per_patch = torch.sum(loss_matrix, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18) 
            loss_student = (loss_student_per_patch * gambling_uncertainty).mean()

        # Auxiliary Cross-Negative Loss (Teacher-Student)
        loss_cross = 0.0
        if teacher_feat is not None:
            # Compute cross-similarity between student and teacher embeddings.
            cross_sim = torch.matmul(feat, teacher_feat.transpose(1, 2)) 
            mem_mask_cross = torch.eq(mask, mask.transpose(1, 2)).float()
            mem_mask_cross_neg = 1 - mem_mask_cross  # Different classes.
            
            # Use a dynamic threshold for teacher negatives instead of selecting top-k negatives.
            cross_neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5)
            cross_hard_neg_mask = mem_mask_cross_neg.bool() & (cross_sim > cross_neg_thresh)
            
            # Compute auxiliary loss for these hard negatives: penalty increases as similarity increases.
            if cross_hard_neg_mask.sum() > 0:
                loss_cross_term = -torch.log(1 - cross_sim + 1e-18)
                loss_cross_term = loss_cross_term * cross_hard_neg_mask.float()
                loss_cross = torch.sum(loss_cross_term) / (torch.sum(cross_hard_neg_mask.float()) + 1e-18)
            else:
                loss_cross = 0.0

        # Total Loss
        total_loss = loss_student + self.lambda_cross * loss_cross
        return total_loss

if __name__ == "__main__":
    # Test the UnCLoss
    s_logits = torch.randn(8, 2, 16, 16, 16)
    t_logits = torch.randn(8, 2, 16, 16, 16)
    beta = 0.8
    uncl = UnCLoss()
    loss = uncl(s_logits, t_logits, beta)
    print(f"uncl_loss: {loss}")
    
    # Test the FeCLoss
    feat = torch.randn(8, 128, 128).cuda()
    mask = torch.randint(0, 2, (8, 1, 128)).cuda()
    decoded_logits = torch.randn(8, 128).cuda()
    
    fecl = FeCLoss(device='cuda:0', use_focal=True)
    loss = fecl(feat=feat, mask=mask, teacher_feat=None, gambling_uncertainty=decoded_logits)
    print(f"fecl_loss: {loss}")
