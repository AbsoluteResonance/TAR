import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=0.0, smooth=1e-5):
        """
        初始化加权Dice Loss。

        参数:
        alpha (float): 控制距离权重的超参数。值越大，边界附近的权重越高。
        gamma (float): 控制焦点损失（focal loss）的超参数。gamma > 0 时，降低易分类样本的权重。
        smooth (float): 平滑项，防止分母为零。
        """
        super(WeightedDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true, distmap):
        """
        计算加权Dice Loss。

        参数:
        y_pred (torch.Tensor): 模型输出的 logits，形状为 (batch_size, 2, depth, height, width)。
        y_true (torch.Tensor): 真实标签，形状为 (batch_size, 1, depth, height, width)。
        distmap (torch.Tensor): 前景的距离图，形状为 (batch_size, 1, depth, height, width)。

        返回:
        torch.Tensor: 计算得到的加权Dice Loss。
        """
        # 将 y_true 转换为 one-hot 编码，形状变为 (batch_size, 2, depth, height, width)
        y_true = y_true.squeeze(1).long()  # 去掉通道维度
        y_true = F.one_hot(y_true, num_classes=2).permute(0, 4, 1, 2, 3).float()  # (B, 2, D, H, W)

        # 对 y_pred 进行 softmax，得到每个类别的概率
        y_pred = F.softmax(y_pred, dim=1)  # (B, 2, D, H, W)

        # 提取前景通道（通道1）的预测概率和真实标签
        y_pred_fg = y_pred[:, 1:, ...]  # (B, 1, D, H, W)
        y_true_fg = y_true[:, 1:, ...]   # (B, 1, D, H, W)

        # 计算权重：距离越近（绝对值小）的像素权重越高
        weights = torch.exp(-self.alpha * distmap.abs())  # (B, 1, D, H, W)

        # 可选：结合焦点损失（Focal Loss）思想，降低易分类样本的权重
        if self.gamma > 0:
            pt = (y_pred_fg * y_true_fg).sum(dim=(2, 3, 4)) / (y_pred_fg.sum(dim=(2, 3, 4)) + 1e-6)
            focal_weight = (1 - pt) ** self.gamma
            weights = weights * focal_weight.view(weights.shape[0], 1, 1, 1, 1)

        # 计算加权Dice系数的分子和分母
        numerator = 2 * torch.sum(weights * y_true_fg * y_pred_fg, dim=(2, 3, 4)) + self.smooth
        denominator = torch.sum(weights * y_true_fg, dim=(2, 3, 4)) + torch.sum(weights * y_pred_fg, dim=(2, 3, 4)) + self.smooth

        # 计算Dice Loss并取平均
        dice = numerator / denominator
        return 1 - dice.mean()

# 示例用法
if __name__ == "__main__":
    # 假设的输入数据
    batch_size = 2
    depth, height, width = 64, 96, 96
    y_pred = torch.randn(batch_size, 2, depth, height, width)  # 模型输出的logits
    y_true = torch.randint(0, 2, (batch_size, 1, depth, height, width))  # 真实标签（0或1）
    distmap = torch.from_numpy(np.load('../dataset/data/Dataset412_aneart/others/distmaps/huashan_lu_52_patch_0_0000.npy')).float()  # 加载距离图

    # 初始化Loss函数
    criterion = WeightedDiceLoss(alpha=1.0, gamma=2.0)  # 可选结合Focal Loss

    # 计算Loss
    loss = criterion(y_pred, y_true, distmap)
    print(f"Weighted Dice Loss: {loss.item()}")