import cv2
import os
import numpy as np
from tqdm import tqdm

def show_masks():
    # 获取当前目录下所有的png文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    png_files = [f for f in os.listdir(current_dir) if f.endswith('.png')]
    
    if not png_files:
        print("当前目录下没有找到PNG文件")
        return
    
    # 创建输出目录
    output_dir = os.path.join(current_dir, 'visualized_masks')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始处理 {len(png_files)} 个mask文件...")
    for png_file in tqdm(png_files):
        # 读取mask图像
        mask_path = os.path.join(current_dir, png_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"无法读取图像: {png_file}")
            continue
        
        # 创建可视化图像（黑色背景，白色缺陷）
        visualization = np.zeros_like(mask)  # 创建全黑背景
        # 将所有非0值（即所有缺陷）设置为白色(255)
        visualization[mask > 0] = 255
        
        # 保存可视化结果
        output_path = os.path.join(output_dir, f'vis_{png_file}')
        cv2.imwrite(output_path, visualization)
    
    print(f"\n处理完成！结果保存在: {output_dir}")

if __name__ == '__main__':
    show_masks()
