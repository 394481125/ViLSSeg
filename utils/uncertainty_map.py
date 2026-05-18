import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label as scipy_label
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def choquet_integral(features, weights):
    # Compute the Choquet integral
    sorted_indices = torch.argsort(features, dim=1, descending=True)
    cum_weights = torch.cumsum(weights[sorted_indices], dim=1)
    integral = torch.sum(features * cum_weights, dim=1)
    return integral

def calculate_mask_area(mask):
    """
    计算mask的面积
    参数:
        mask (torch.Tensor): 包含图像mask的张量，形状为 (b, c, h, w)，其中b是batch大小，c是通道数，h和w是图像的高度和宽度。
    返回:
        torch.Tensor: mask的面积，形状为 (b,)
    """
    # 将mask张量中非零值设置为1
    binary_mask = (mask != 0).float()
    # 计算每个mask的面积
    area = torch.sum(binary_mask, dim=(2, 3))
    return area


def calculate_avg_bbox_area(mask):
    """
    对 batch 中每个样本，用连通域分析找到所有独立目标，
    分别计算每个目标的 bbox 面积，返回平均值。
    适配 MoNuSeg 等密集多目标场景。
    参数:
        mask (torch.Tensor): 形状为 (b, 1, h, w)，值为 0 或 1 的二值 mask
    返回:
        list[float]: 长度为 b，每个元素是该样本的平均单目标 bbox 面积
    """
    B = mask.shape[0]
    avg_areas = []

    for b in range(B):
        # 取出单张图，转为 numpy，shape: (h, w)
        binary = mask[b, 0].detach().cpu().numpy().astype(np.uint8)

        # 连通域标注，labeled 中每个连通域有唯一整数 id，num_features 是连通域数量
        labeled, num_features = scipy_label(binary)

        if num_features == 0:
            avg_areas.append(1.0)  # 避免除零，返回最小面积
            continue

        areas = []
        for comp_id in range(1, num_features + 1):
            # 找到当前连通域的所有像素坐标
            ys, xs = np.where(labeled == comp_id)
            # 计算该连通域的 bbox 面积
            bbox_w = xs.max() - xs.min() + 1
            bbox_h = ys.max() - ys.min() + 1
            areas.append(float(bbox_w * bbox_h))

        avg_areas.append(np.mean(areas))

    return avg_areas


def compute_uncertainty(output, weight_um):
    # Reshape output to (B, C, H*W) for computation
    B, C, H, W = output.shape
    output_reshaped = output.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)

    probabilities = F.softmax(output_reshaped, dim=1)
    num_classes = probabilities.size(1)
    features = probabilities.view(B * H * W, -1)

    # weight_um = torch.linspace(0, 1, num_classes, device='cuda:0')
    integral = choquet_integral(features, weight_um)

    # Normalize integral
    integral = (integral - integral.min()) / (integral.max() - integral.min())

    # Reshape integral back to (B, H, W)
    uncertainty_mask = integral.view(B, 1, H, W)

    mask_area = calculate_mask_area(uncertainty_mask)  # shape: (B, 1)
    avg_bbox_areas = calculate_avg_bbox_area(uncertainty_mask)  # list, 长度 B

    # 遍历每个批次，用各自的平均单目标 bbox 面积计算 threshold
    for batch_idx in range(B):
        current_uncertainty_mask = uncertainty_mask[batch_idx]  # (1, H, W)
        current_mask_area = mask_area[batch_idx]  # (1,)
        current_bbox_area = torch.tensor(avg_bbox_areas[batch_idx],
                                         device=uncertainty_mask.device)

        # threshold 含义不变：mask 紧密程度，值越接近 0.5 说明目标越紧凑
        current_threshold = (current_mask_area + current_bbox_area) / (2 * current_bbox_area)

        current_uncertainty_mask = torch.where(
            current_uncertainty_mask >= current_threshold, 1.0, 0.0
        )
        uncertainty_mask[batch_idx] = current_uncertainty_mask

    return uncertainty_mask


