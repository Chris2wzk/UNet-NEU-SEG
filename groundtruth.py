#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

def convert_segmentation(segmentation, color_map):
    """
    将输入的分割图转换为单通道类别索引（如果需要，使用 argmax 操作）并映射成彩色图
    """
    # 如果图像是多通道，则沿着通道维度使用 argmax 得到类别索引
    if segmentation.ndim == 3 and segmentation.shape[2] > 1:
        seg_index = np.argmax(segmentation, axis=2)
    elif segmentation.ndim == 2:
        seg_index = segmentation
    else:
        raise ValueError("不支持的分割图像数据格式")
    
    h, w = seg_index.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in color_map.items():
        colored[seg_index == cls_id] = color
    return seg_index, colored

def process_segmentation_dir(seg_dir, output_dir, color_map):
    """
    处理指定目录下所有 png 分割文件，转换为单通道类别索引及彩色图，并保存到输出目录
    """
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"分割标注源路径不存在: {seg_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历目录下所有 png 文件
    files = [f for f in os.listdir(seg_dir) if f.lower().endswith('.png')]
    if not files:
        print("在指定目录下未找到 png 文件")
        return
    
    for file in files:
        seg_path = os.path.join(seg_dir, file)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg is None:
            print(f"[WARN] 无法读取文件: {seg_path}")
            continue
        
        try:
            seg_index, colored = convert_segmentation(seg, color_map)
        except Exception as e:
            print(f"[ERROR] 处理文件 {seg_path} 时发生错误: {str(e)}")
            continue
        
        base_name = os.path.splitext(file)[0]
        index_save_path = os.path.join(output_dir, base_name + "_index.png")
        colored_save_path = os.path.join(output_dir, base_name + "_colored.png")
        
        cv2.imwrite(index_save_path, seg_index)
        cv2.imwrite(colored_save_path, colored)
        print(f"处理完成: {seg_path}\n 生成单通道索引: {index_save_path}\n 彩色结果: {colored_save_path}")

def main():
    # 在 main 函数中直接设置路径，不通过命令行参数传递
    segmentation_source_dir = "data_wenjian/SegmentationClass"  # 分割标注文件目录，根据实际情况修改
    output_dir = "colored_segmentation_results"  # 保存结果的目录

    # 定义颜色映射（BGR 格式）
    color_map = {
        0: (0, 0, 0),       # 背景：黑色
        1: (0, 0, 255),     # 缺陷1：红色
        2: (255, 0, 0),     # 缺陷2：蓝色
        3: (0, 255, 0)      # 缺陷3：绿色
    }
    
    process_segmentation_dir(segmentation_source_dir, output_dir, color_map)

if __name__ == '__main__':
    main()