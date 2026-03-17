import torch, nibabel as nib
import numpy as np

torch.set_default_dtype(torch.float32)
device = torch.device(f"cuda:0")

def Dice(outputs_binary, labels, smooth=1e-5, dice_one=True, images=None, erase_background=False):
    if erase_background:
        outputs_binary[images == 0] = 0

    intersection = (outputs_binary * labels).sum(dim=(1, 2, 3, 4))

    outputs_sum = outputs_binary.sum(dim=(1, 2, 3, 4))
    labels_sum = labels.sum(dim=(1, 2, 3, 4))

    dice_scores = []
    for i in range(outputs_binary.size(0)):
        if labels_sum[i] == 0:
            if outputs_sum[i] == 0:
                if dice_one:
                    dice_scores.append(1.0)
            else:
                dice_scores.append(0.0)
        else:
            dice = (2.0 * intersection[i] + smooth) / (outputs_sum[i] + labels_sum[i] + smooth)
            dice_scores.append(dice.item())

    return torch.tensor(dice_scores).sum().item()

def multilabel_Dice(output, label, smooth=1e-5, dice_one=True, weights=None):
    output = output.squeeze(1)
    label = label.squeeze(1)
    B = output.shape[0]
    
    # 计算整个batch中的最大类别
    if label.numel() == 0:
        max_class = 0
    else:
        max_class = int(torch.max(label).item())
    
    all_dices = [0.0] * (max_class + 1)
    class_counts = [0] * (max_class + 1)
    
    for b in range(B):
        output_sample = output[b]
        label_sample = label[b]
        
        for cls in range(max_class + 1):
            mask_label = (label_sample == cls).int()
            sum_l = torch.sum(mask_label).item()
            
            mask_output = (output_sample == cls).int()
            sum_o = torch.sum(mask_output).item()
            
            if sum_l > 0:
                # 标签中存在该类别，正常计算Dice
                intersection = torch.sum(mask_output * mask_label).item()
                dice = (2.0 * intersection + smooth) / (sum_o + sum_l + smooth)
            else:
                # 标签中不存在该类别
                if sum_o > 0:
                    # 输出中存在该类别，Dice为0
                    dice = 0.0
                else:
                    # 输出中也不存在该类别，根据dice_one处理
                    if dice_one:
                        dice = 1.0
                    else:
                        # 跳过该类别在该样本中的计算
                        continue
            
            # 累加Dice值和计数
            all_dices[cls] += dice
            class_counts[cls] += 1
    
    # 计算各类别平均Dice
    for cls in range(len(all_dices)):
        if class_counts[cls] > 0:
            all_dices[cls] /= class_counts[cls]
    
    # 收集有效类别（非背景且被统计过的类别）
    valid_dices = []
    for cls in range(1, len(all_dices)):
        if class_counts[cls] > 0:
            valid_dices.append(all_dices[cls])
    
    return np.mean(valid_dices) if valid_dices else 0.0

import cupy as cp
from cupyx.scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
def points_Dice(pred_points, true_points, threshold=cp.array([10])):
    """
    计算关键点的Dice指标
    :param pred_points: 预测的关键点，形状为 (num_pred, 2) 或 (num_pred, 3)
    :param true_points: 真实的关键点，形状为 (num_true, 2) 或 (num_true, 3)
    :param threshold: 匹配阈值
    :return: Dice指标
    """
    if len(pred_points) == 0 and len(true_points) == 0:
        return 1.0
    elif len(pred_points) == 0 or len(true_points) == 0:
        return 0.0

    # 计算预测点和真实点之间的距离
    distances = cdist(pred_points, true_points)

    # 使用匈牙利算法进行匹配
    row_ind, col_ind = linear_sum_assignment(distances)

    # 找到距离小于阈值的匹配
    matches = distances[row_ind, col_ind] < threshold[col_ind]
    tp = matches.sum()
    fp = len(pred_points) - tp
    fn = len(true_points) - tp

    # 计算Dice
    dice = tp / (tp + fp + fn + 1e-8)
    return dice