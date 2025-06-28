import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple

def binary_to_multiclass_inference(binary_mask, thresholds=[0.3, 0.5, 0.7, 0.8, 0.9]):
    """
    在推理阶段，将二值掩码预测概率转换为多个类别
    
    参数:
    - binary_mask: 模型预测的二值掩码，值在[0,1]范围，表示缺陷概率
    - thresholds: 不同类别的概率阈值，用于区分不同缺陷类型的置信度级别
    
    返回:
    - multi_mask: 多类别掩码，值范围为[0,1,2,3,4,5]
                 0表示背景，1-5表示五种不同类型的缺陷
    """
    # 初始化为全0背景
    multi_mask = torch.zeros_like(binary_mask, dtype=torch.long)
    
    # 根据不同阈值划分为不同类别
    multi_mask[binary_mask > thresholds[4]] = 5  # 最高置信度 - 类别5
    multi_mask[(binary_mask > thresholds[3]) & (binary_mask <= thresholds[4])] = 4  # 类别4
    multi_mask[(binary_mask > thresholds[2]) & (binary_mask <= thresholds[3])] = 3  # 类别3
    multi_mask[(binary_mask > thresholds[1]) & (binary_mask <= thresholds[2])] = 2  # 类别2
    multi_mask[(binary_mask > thresholds[0]) & (binary_mask <= thresholds[1])] = 1  # 类别1
    
    return multi_mask

def classify_by_morphology(binary_mask: torch.Tensor) -> torch.Tensor:
    """
    基于形态学特征将二值掩码中的缺陷区域分类为多个类别
    
    参数:
    - binary_mask: 二值掩码，值为0或1
    
    返回:
    - multi_mask: 多类别掩码，值范围为[0,1,2,3,4,5]
    """
    # 转换为numpy数组进行形态学处理
    device = binary_mask.device
    mask_np = binary_mask.cpu().numpy().astype(np.uint8)
    
    # 寻找所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
    # 初始化多类别掩码
    multi_mask = np.zeros_like(mask_np)
    
    # 跳过背景(索引0)
    for i in range(1, num_labels):
        # 获取当前连通区域的面积和形状特征
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # 计算周长和圆形度
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        perimeter = cv2.arcLength(contours[0], True)
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        aspect_ratio = float(width) / height if height > 0 else 0
        
        # 根据形态特征分类
        if area < 50:  # 小面积缺陷
            defect_class = 1
        elif circularity > 0.8:  # 高圆形度
            defect_class = 2
        elif aspect_ratio > 3 or aspect_ratio < 0.33:  # 细长形状
            defect_class = 3
        elif area > 1000:  # 大面积缺陷
            defect_class = 4
        else:  # 其他形状
            defect_class = 5
            
        # 在多类别掩码中标记
        multi_mask[labels == i] = defect_class
    
    return torch.tensor(multi_mask, device=device)

def dynamic_thresholding(pred_prob, num_classes=6, min_threshold=0.2):
    """
    动态确定阈值，而不是使用固定阈值
    
    参数:
    - pred_prob: 预测的概率图
    - num_classes: 目标类别数量（包括背景）
    - min_threshold: 最小阈值
    
    返回:
    - multi_mask: 根据动态阈值分类的多类别掩码
    """
    device = pred_prob.device
    # 使用k-means等聚类算法划分值域
    # 为了简化，这里使用等间隔划分
    step = (1.0 - min_threshold) / (num_classes - 1)
    thresholds = [min_threshold + step * i for i in range(num_classes - 1)]
    
    # 初始化结果掩码
    multi_mask = torch.zeros_like(pred_prob, dtype=torch.long)
    
    # 应用阈值
    for i in range(num_classes - 1):
        if i == 0:
            # 第一个类别：大于最小阈值，小于等于第二个阈值
            multi_mask[(pred_prob > thresholds[i]) & (pred_prob <= thresholds[i+1] if i+1 < len(thresholds) else 1.1)] = i + 1
        elif i == len(thresholds) - 1:
            # 最后一个类别：大于最后一个阈值
            multi_mask[pred_prob > thresholds[i]] = i + 1
        else:
            # 中间类别：大于当前阈值，小于等于下一个阈值
            multi_mask[(pred_prob > thresholds[i]) & (pred_prob <= thresholds[i+1])] = i + 1
    
    return multi_mask

def binary_mask_to_multiclass(inputs, targets, training=True):
    """
    训练过程中处理二值掩码数据，便于多通道输出训练
    
    参数:
    - inputs: 模型输出，形状为[B, C, H, W]，C为类别数(6)
    - targets: 目标掩码，形状为[B, H, W]，值为0或1
    - training: 是否在训练模式
    
    返回:
    - loss_targets: 计算损失用的目标，形状为[B, H, W]
    """
    # 检查targets是否为二值掩码
    unique_vals = torch.unique(targets)
    is_binary = len(unique_vals) <= 2 and torch.max(unique_vals) <= 1
    
    if not is_binary:
        # 如果不是二值掩码，直接返回原始目标
        return targets
        
    if training:
        # 训练模式下，对二值掩码进行特殊处理
        # 创建等价的多类别掩码用于损失计算
        # 所有前景像素（原值为1）都算作类别1
        # 这样模型仍然学习多类别输出，但损失计算只区分前景/背景
        return targets
    else:
        # 推理模式下，使用概率阈值划分多类别
        probs = F.softmax(inputs, dim=1)
        # 取所有前景类（索引1-5）的最大概率
        max_probs, pred_classes = torch.max(probs[:, 1:], dim=1)
        # 索引加1，因为我们只考虑了前景类
        pred_classes = pred_classes + 1
        
        # 创建预测掩码：背景为0，前景为预测的类别
        preds = torch.zeros_like(targets)
        mask = max_probs > 0.5  # 前景概率阈值
        preds[mask] = pred_classes[mask]
        
        return preds 