import torch
import torch.nn.functional as F

def cl_score(v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """计算骨架体积重叠分数 (GPU加速版本)
    
    Args:
        v (torch.Tensor): 二值图像 (bool或0/1)
        s (torch.Tensor): 骨架图像 (bool或0/1)
        
    Returns:
        torch.Tensor: 计算得到的骨架体积重叠分数
    """
    return torch.sum(v * s) / torch.sum(s)

def clDice(v_p: torch.Tensor, v_l: torch.Tensor) -> torch.Tensor:
    """计算cldice指标 (GPU加速版本)
    
    Args:
        v_p (torch.Tensor): 预测图像 (batch_size, 1, D, H, W)
        v_l (torch.Tensor): 真实标签图像 (batch_size, 1, D, H, W)
        
    Returns:
        torch.Tensor: cldice指标值
    """
    # 确保输入在GPU上
    device = v_p.device
    
    # 如果输入是batch形式，取第一个样本
    if v_p.dim() == 5:
        v_p = v_p[0, 0]  # 取第一个样本的第一个通道
    if v_l.dim() == 5:
        v_l = v_l[0, 0]
    
    # 转换为布尔类型
    v_p_bool = (v_p > 0.5).float()
    v_l_bool = (v_l > 0.5).float()
    
    # 3D骨架化
    skeleton_l = skelexon3d(v_l_bool.unsqueeze(0).unsqueeze(0)).squeeze()
    skeleton_p = skelexon3d(v_p_bool.unsqueeze(0).unsqueeze(0)).squeeze()
    
    # 计算tprec和tsens
    tprec = cl_score(v_p_bool, skeleton_l)
    tsens = cl_score(v_l_bool, skeleton_p)
    
    # 计算clDice
    return 2 * tprec * tsens / (tprec + tsens + 1e-8)  # 添加小量防止除以0