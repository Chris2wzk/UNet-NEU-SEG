import os
import shutil
import glob  # 新增glob模块

def extract_files(txt_file, src_jpeg, src_mask, dst_jpeg, dst_mask):
    # 创建目标目录（如果不存在的话）
    if not os.path.exists(dst_jpeg):
        os.makedirs(dst_jpeg)
    if not os.path.exists(dst_mask):
        os.makedirs(dst_mask)
    
    with open(txt_file, 'r') as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue

            # 针对JPEGImages文件夹，利用glob匹配所有扩展名的文件
            pattern_jpeg = os.path.join(src_jpeg, filename + ".*")
            jpeg_files = glob.glob(pattern_jpeg)
            if not jpeg_files:
                print(f"Warning: No file matching {pattern_jpeg} found in {src_jpeg}!")
            else:
                for file_path in jpeg_files:
                    shutil.copy(file_path, dst_jpeg)
                    print(f"Copied {file_path} to {dst_jpeg}")
                if len(jpeg_files) > 1:
                    print("--------- 分割线 ---------")

            # 针对SegmentationClass文件夹，利用glob匹配所有扩展名的文件  
            pattern_mask = os.path.join(src_mask, filename + ".*")
            mask_files = glob.glob(pattern_mask)
            if not mask_files:
                print(f"Warning: No file matching {pattern_mask} found in {src_mask}!")
            else:
                for file_path in mask_files:
                    shutil.copy(file_path, dst_mask)
                    print(f"Copied {file_path} to {dst_mask}")
                if len(mask_files) > 1:
                    print("--------- 分割线 ---------")

def main():
    # 设置三个路径（txt文件和两个源文件夹）
    txt_file = "/data/NEU_SEG/data_wenjian/ImageSets/Segmentation/val.txt"
    src_jpeg = "/data/NEU_SEG/data_wenjian/JPEGImages"
    src_mask = "/data/NEU_SEG/data_wenjian/SegmentationClass"
    
    # 定义目标文件夹（这里采用在原文件夹基础上添加后缀 _selected，也可以根据需要修改）
    dst_jpeg = "img_dir/val"
    dst_mask = "ann_dir/val"
    
    extract_files(txt_file, src_jpeg, src_mask, dst_jpeg, dst_mask)

if __name__ == "__main__":
    main()
