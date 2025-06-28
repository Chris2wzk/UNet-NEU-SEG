import os
import random

def rename_jpegs_in_current_directory():
    # 获取当前路径
    current_path = os.getcwd()

    # 找到所有以 .jpg 结尾的文件
    jpg_files = [f for f in os.listdir(current_path) if f.lower().endswith('.jpg')]

    # 计算文件总数
    total_count = len(jpg_files)
    print(f"共找到 {total_count} 个 .jpg 文件。")

    if total_count == 0:
        print("没有找到任何 .jpg 文件，程序结束。")
        return

    # 打乱顺序
    random.shuffle(jpg_files)

    # 按照新的命名规则进行重命名
    # 格式：0001.jpg ~ total_count.jpg
    for idx, old_name in enumerate(jpg_files, start=1):
        new_name = f"{idx:04d}.jpg"  # 生成形如 0001.jpg 的文件名
        old_path = os.path.join(current_path, old_name)
        new_path = os.path.join(current_path, new_name)
        os.rename(old_path, new_path)
        print(f"重命名：{old_name} => {new_name}")

if __name__ == "__main__":
    rename_jpegs_in_current_directory() 