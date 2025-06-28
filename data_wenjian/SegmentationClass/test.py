def check_label_distribution(label_dir="path/to/label_masks", file_extension=".png"):
    """
    批量读取标签图像并打印其像素值分布。
    :param label_dir: 存放标签图像的目录路径
    :param file_extension: 标签图像的后缀 (如 '.png', '.jpg'等)
    """
    import os
    from PIL import Image
    import numpy as np

    # 列出目录下全部以指定后缀结尾的文件
    label_files = [f for f in os.listdir(label_dir) if f.endswith(file_extension)]

    # 打印找到的标签文件数量
    print(f"在 {label_dir} 中找到了 {len(label_files)} 个标签文件。")

    # 逐个读取并打印 unique 像素值（这里仅示例前 5 个，可自行调整或去掉）
    for idx, filename in enumerate(label_files):
        if idx >= 1000:
            break
        filepath = os.path.join(label_dir, filename)

        # 打开图像并转换为 NumPy 数组
        label_img = Image.open(filepath)
        label_np = np.array(label_img)

        # 获取 unique 像素值
        unique_vals = np.unique(label_np)
        print(f"文件名: {filename}, unique 像素值: {unique_vals}")

if __name__ == "__main__":
    # 修改此处为你的标签图像目录和需要处理的图像后缀名
    check_label_distribution(label_dir="/data/NEU_Seg__unet-main/data_wenjian/SegmentationClass", file_extension=".png")