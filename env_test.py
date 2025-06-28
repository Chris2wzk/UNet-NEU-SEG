#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境测试脚本
用于验证PyTorch和CUDA是否正确安装
"""

import sys
import torch
import torchvision

def test_pytorch():
    """测试PyTorch安装"""
    print("=" * 50)
    print("PyTorch环境测试")
    print("=" * 50)
    
    # 基本信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Torchvision版本: {torchvision.__version__}")
    
    # CUDA信息
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA不可用，将使用CPU进行训练")
    
    # 测试基本操作
    print("\n测试基本张量操作...")
    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("✓ 基本张量操作正常")
    except Exception as e:
        print(f"✗ 基本张量操作失败: {e}")
        return False
    
    # 测试CUDA操作（如果可用）
    if torch.cuda.is_available():
        print("\n测试CUDA操作...")
        try:
            x_cuda = torch.randn(3, 3).cuda()
            y_cuda = torch.randn(3, 3).cuda()
            z_cuda = torch.mm(x_cuda, y_cuda)
            print("✓ CUDA操作正常")
        except Exception as e:
            print(f"✗ CUDA操作失败: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("环境测试完成！")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_pytorch()
    if success:
        print("🎉 环境配置成功！可以开始训练了。")
    else:
        print("❌ 环境配置有问题，请检查安装。")
        sys.exit(1) 