import math

import numpy as np
import scipy.stats
import torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import torch
from utils.mask_gen import MaskGenerator

class BoxMaskGenerator3D(MaskGenerator):
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=True, 
                 prop_by_volume=True, within_bounds=True, invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_volume = prop_by_volume  # 改为体积比例
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        生成3D立方体掩码参数
        :param n_masks: 掩码数量（批次大小）
        :param mask_shape: 掩码形状 (H, W, D)
        :param rng: [可选] np.random.RandomState实例
        :return: 掩码数组 (N, 1, H, W, D)
        """
        if rng is None:
            rng = np.random
            
        H, W, D = mask_shape  # 三维形状
        
        if self.prop_by_volume:
            # 生成立方体体积比例
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], 
                                    size=(n_masks, self.n_boxes))
            
            zero_mask = mask_props == 0.0
            
            if self.random_aspect_ratio:
                # 三维随机比例
                h_props = np.exp(rng.uniform(0, np.log(mask_props), 
                                           size=mask_props.shape))
                w_props = np.exp(rng.uniform(0, np.log(mask_props), 
                                           size=mask_props.shape))
                d_props = mask_props / (h_props * w_props)
                h_props *= np.cbrt(1.0 / self.n_boxes)
                w_props *= np.cbrt(1.0 / self.n_boxes)
                d_props *= np.cbrt(1.0 / self.n_boxes)
            else:
                # 等比例立方体
                scale = np.cbrt(mask_props * 1.0 / self.n_boxes)
                h_props = w_props = d_props = scale
        else:
            # 直接生成各维度比例
            if self.random_aspect_ratio:
                h_props = rng.uniform(*self.prop_range, 
                                     size=(n_masks, self.n_boxes))
                w_props = rng.uniform(*self.prop_range, 
                                     size=(n_masks, self.n_boxes))
                d_props = rng.uniform(*self.prop_range, 
                                     size=(n_masks, self.n_boxes))
                scale = np.cbrt(1.0 / self.n_boxes)
                h_props *= scale
                w_props *= scale
                d_props *= scale
            else:
                size = rng.uniform(*self.prop_range, 
                                  size=(n_masks, self.n_boxes))
                scale = np.cbrt(size * 1.0 / self.n_boxes)
                h_props = w_props = d_props = scale

        # 计算实际尺寸（四舍五入取整）
        h_sizes = np.round(h_props * H).astype(int)
        w_sizes = np.round(w_props * W).astype(int)
        d_sizes = np.round(d_props * D).astype(int)
        
        # 生成立方体位置
        if self.within_bounds:
            h_starts = np.round(rng.uniform(0, H - h_sizes, size=h_sizes.shape)).astype(int)
            w_starts = np.round(rng.uniform(0, W - w_sizes, size=w_sizes.shape)).astype(int)
            d_starts = np.round(rng.uniform(0, D - d_sizes, size=d_sizes.shape)).astype(int)
            h_ends = h_starts + h_sizes
            w_ends = w_starts + w_sizes
            d_ends = d_starts + d_sizes
        else:
            centers_h = np.round(rng.uniform(0, H, size=h_sizes.shape)).astype(int)
            centers_w = np.round(rng.uniform(0, W, size=w_sizes.shape)).astype(int)
            centers_d = np.round(rng.uniform(0, D, size=d_sizes.shape)).astype(int)
            h_starts = np.maximum(0, centers_h - h_sizes // 2)
            w_starts = np.maximum(0, centers_w - w_sizes // 2)
            d_starts = np.maximum(0, centers_d - d_sizes // 2)
            h_ends = np.minimum(H, centers_h + h_sizes // 2)
            w_ends = np.minimum(W, centers_w + w_sizes // 2)
            d_ends = np.minimum(D, centers_d + d_sizes // 2)
        
        # 初始化掩码
        if self.invert:
            masks = np.zeros((n_masks, 1, H, W, D), dtype=np.float32)
        else:
            masks = np.ones((n_masks, 1, H, W, D), dtype=np.float32)
            
        # 填充立方体区域
        for i in range(n_masks):
            for box_idx in range(self.n_boxes):
                h0, h1 = h_starts[i, box_idx], h_ends[i, box_idx]
                w0, w1 = w_starts[i, box_idx], w_ends[i, box_idx]
                d0, d1 = d_starts[i, box_idx], d_ends[i, box_idx]
                
                # 翻转立方体区域的值
                masks[i, 0, h0:h1, w0:w1, d0:d1] = 1 - masks[i, 0, h0:h1, w0:w1, d0:d1]
                
        return masks

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        # 保持与原接口兼容
        return t_params
    
class CutMix3D(object):
    def __init__(self, mix_ratio=0.5,
                 mask_prop_range=0.5, 
                 boxmask_n_boxes=1, 
                 boxmask_fixed_aspect_ratio=False, 
                 boxmask_by_size=False, 
                 boxmask_outside_bounds=False, 
                 boxmask_no_invert=False):
        self.mask_generator = BoxMaskGenerator3D(prop_range=mask_prop_range, n_boxes=boxmask_n_boxes,
                                               random_aspect_ratio=not boxmask_fixed_aspect_ratio,
                                               prop_by_volume=not boxmask_by_size, within_bounds=not boxmask_outside_bounds,
                                               invert=not boxmask_no_invert)
        self.mix_ratio = mix_ratio
    
    def apply_cutmix(self, images, labels):
        """
        对一组有监督数据进行CutMix混合，随机选择部分数据进行混合
        
        参数:
            images (torch.Tensor): 输入图像 [N, C, H, W, D]
            labels (torch.Tensor): 输入标签 [N, H, W, D] (整数标签)
            mix_ratio (float): 进行CutMix的数据比例 (0.0~1.0)
        
        返回:
            mixed_images: 混合后的图像 [N, C, H, W, D]
            mixed_labels: 混合后的标签 [N, H, W, D]
            masks: 使用的混合掩码 [N, 1, H, W, D]
        """
        device = images.device
        N, C, H, W, D = images.shape
        
        # 确定需要混合的样本数量
        mix_n = int(N * 0.5 * self.mix_ratio)
        if mix_n == 0:  # 如果没有样本需要混合，直接返回原始数据
            return images, labels, torch.ones((N, 1, H, W, D), device=device)
        
        # 随机选择进行CutMix的样本索引
        mix_indices = torch.randperm(N)[:mix_n]
        available_indices = list(set(range(N)) - set(mix_indices.tolist()))
        pair_indices = torch.randperm(len(available_indices))[:mix_n]
        
        # 初始化输出
        mixed_images = images.clone()
        mixed_labels = labels.clone()
        masks = torch.ones((N, 1, H, W, D), device=device)
        
        # 为混合样本生成掩码参数
        mask_params = self.mask_generator.generate_params(mix_n, (H, W, D))
        mask_params_tensor = torch.tensor(mask_params, dtype=torch.float32, device=device)
        
        # 批量生成掩码 [mix_n, 1, H, W, D]
        mix_masks = self.mask_generator.torch_masks_from_params(mask_params_tensor, (H, W, D), device)
        
        # 对每个混合样本应用CutMix
        for i, mix_idx in enumerate(mix_indices):
            pair_idx = pair_indices[i]
            
            # 应用混合
            mixed_images[mix_idx] = images[mix_idx] * mix_masks[i] + images[pair_idx] * (1 - mix_masks[i])
            
            # 扩展掩码维度以匹配标签
            mask_expanded = mix_masks[i].squeeze(0)  # [H, W, D]

            # 混合标签
            if mask_expanded[H//2, W//2, D//2] > 0.5:
                mixed_labels[mix_idx] = torch.where(mask_expanded > 0.5, labels[mix_idx], 0.)
            else:
                mixed_labels[mix_idx] = torch.where(mask_expanded > 0.5, 0., labels[pair_idx])
            #mixed_labels[mix_idx] = torch.where(mask_expanded > 0.5, labels[mix_idx], labels[pair_idx])

            masks[mix_idx] = mix_masks[i]
        
        return mixed_images, mixed_labels, masks