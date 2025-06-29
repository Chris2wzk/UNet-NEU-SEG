# NEU-SEG: 钢材表面缺陷语义分割项目

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.6+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**基于UNet的钢材表面缺陷语义分割系统**

[English](README.md) | 中文

</div>

## 📖 项目简介

- 本项目是一个基于UNet进行钢材表面缺陷分割，支持多种改进模型。目前使用的数据集是NEU-SEG数据集，使用NEU-DET数据集进行目标检测的朋友，可以用这个同系列数据集进行补充工作量。项目实现了从基线UNet到各种改进版本的完整训练、验证和推理流程，是进行钢材缺陷检测研究的理想起点。

- 强烈建议使用`Cursor`进行项目梳理。

## 📞 联系方式

- **邮箱**: 1318489612@qq.com
⭐ **如果这个项目对你有帮助，请给个Star支持一下！** ⭐

### ✨ 主要特性

- 🎯 **多模型支持**: 包含UNet基线模型及多种模块改进

- 🔧 **模块化设计**: 易于扩展和自定义新的改进方法

- 📊 **完整评估**: 支持多种评估指标（IoU、Dice、准确率等），主要使用mIoU进行评估

- 🖼️ **可视化输出**: 可以可视化分割结果

  

### 🏆 支持的模型

模块都存放在blocks/下

具体的改进模型在unet\unet_model.py和unet\unet_parts.py

可以仿照风格调用blocks文件夹中进行改进模型



## 🚀 快速开始

### 环境要求

本人使用NVIDIA 4090 24GB进行训练，以下是最基础的配置

- Python 3.9+
- PyTorch 1.13+
- CUDA 11.6+ (GPU训练)
- 8GB+ RAM

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/NEU-SEG.git
cd NEU-SEG
```

2. **创建虚拟环境**
```bash
conda create -n neu-seg python=3.9.7
conda activate neu-seg
```

3. **安装PyTorch**
```bash
# CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```
也可以通过whl文件安装：

进入pytorch官网根据cuda号和python版本选择对应的whl进行下载，我使用的是[torch-1.13.1+cu116-cp39-cp39-win_amd64.whl](https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-win_amd64.whl#sha256=80a6b55915ac72c087ab85122289431fde5c5a4c85ca83a38c6d11a7ecbfdb35)

还有[torchvision-0.14.1+cu116-cp310-cp310-linux_x86_64.whl](https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp39-cp39-win_amd64.whl#sha256=4b75cfe80d1e778f252fce94a7dd4ea35bc66a10efd53c4c63910ee95425face)

进入虚拟环境后，切换到whl的下载路径，然后

```
pip install your_package.whl
```

这里的your_package需要替换成具体你下载的whl名称

4. **安装依赖包**
```bash
pip install -r requirements.txt
pip install timm tensorboardX opencv-python pillow tqdm
```

5. **验证安装**
```bash
python env_test.py
```

安装成功会打印出类似日志如下：

```cmd
PyTorch version: 1.13.1+cu116
Is CUDA available? True
CUDA version PyTorch was built with: 11.6
Number of GPUs available: 1
GPU Name: NVIDIA GeForce RTX 4090 GPU
```

### 数据集准备

#### NEU-SEG数据集
```bash
# 数据集结构
data_wenjian/
├── img_dir/
│   ├── train/     # 训练图像
│   ├── val/       # 验证图像
│   └── test/      # 测试图像
├── ann_dir/
│   ├── train/     # 训练标注
│   ├── val/       # 验证标注
│   └── test/      # 测试标注
└── ImageSets/
    └── Segmentation/
        ├── train.txt
        ├── val.txt
        └── test.txt
```



## 🎯 使用方法

### 训练模型

1. **单模型训练**
```bash
python train.py \
    --dataset pascal \
    --model_type UNet \
    --batch_size 8 \
    --epochs 75 \
```

2. **多模型批量训练**

可以通过sh脚本来控制训练的不同模型、不同的轮次，实现单次串行训练多个模型，

每个模型训练完以后自动在测试集上进行validate

### 验证模型

```bash
python val.py 
```

### 推理预测

```bash
python predict.py 
```

推理结束后，会生成`result_unet`文件夹，存放推理之后高亮显示的mask文件

## 📊 数据集说明

### NEU-SEG数据集
- **来源**: 东北大学钢材表面缺陷数据集
- **类别**: 4类（背景、夹渣、斑块、划痕）
- **图像数量**: 约4500张
- **分辨率**: 200×200像素
- **用途**: 主要训练和测试数据集

### 部分数据集示例
![image](https://github.com/user-attachments/assets/8f36ae17-d8fe-46ca-8b9b-389fa5b3b335){width=50%}



## 🏗️ 模型架构

### 基线UNet架构
![image](https://github.com/user-attachments/assets/b04934fe-35ca-4415-b471-103803d6bd0a){width=25%}
本人毕业论文中使用的UNet_RepLKBlock_FastKAN架构
![image](https://github.com/user-attachments/assets/fa695e29-9717-42f9-ad9d-4157bc96e588){width=25%}


### 改进方法

1. **卷积层改进**
   - RepLKBlock: 大核卷积，提升感受野
   - PConv: 并行卷积，减少计算量
   - ODConv: 动态卷积，增强特征表达

2. **注意力机制**
   - CBAM: 通道和空间注意力
   - SimAM: 简单注意力模块
   - ShuffleAttention: 通道注意力
   - FastKAN: 快速核注意力

3. **骨干网络替换**
   - MobileNet系列: 轻量化设计
   - ResNet: 残差连接
   - 自定义改进模块

## 📈 实验结果

### 性能对比
| 模型 | mIoU (%) | mPA(%) | Dice |
|------|---------|----------|------------|
| UNet | 80.6     | 90.1     | 93.7     |
| UNet_RepLKBlock | 83.4     | 91.2     | 95.3     |
| UNetFastKAN | 83.8     | 91.0     | 95.2     |
| UNet_RepLKBlock_FastKAN | **84.8** | **92.3** | **95.6** |

### 基线模型和改进模型训练曲线
![image](https://github.com/user-attachments/assets/3a9eae1f-a63e-456c-937d-bf44f39090b2){width=50%}




## 🔧 配置说明

### 主要参数
```python
# 数据集配置
--dataset pascal          # 数据集名称 (pascal/mydataset)
--batch_size 8           # 批次大小
--crop_size 512          # 裁剪尺寸

# 模型配置
--model_type UNet        # 模型类型
--n_channels 3           # 输入通道数
--n_classes 4            # 类别数

# 训练配置
--epochs 100             # 训练轮数
--lr 0.01               # 学习率
--momentum 0.9          # 动量
--weight_decay 5e-4     # 权重衰减

# 损失函数
--loss_type ce          # 损失类型 (ce/focal/dice)
--use_balanced_weights  # 是否使用类别权重

# 优化器配置
--lr_scheduler poly     # 学习率调度器
--nesterov             # 是否使用Nesterov动量
```

## 📁 项目结构

```
NEU_SEG/
├── blocks/                 # 改进模块实现
│   ├── RepLKBlock.py      # 大核卷积模块
│   ├── FastKan.py         # 快速注意力模块
│   ├── CBAM.py           # CBAM注意力模块
│   └── ...
├── dataloaders/           # 数据加载器
│   ├── datasets/         # 数据集类
│   ├── custom_transforms.py  # 数据增强
│   └── utils.py          # 数据工具
├── unet/                  # UNet模型实现
│   ├── unet_model.py     # 模型定义
│   ├── unet_parts.py     # 模型组件
│   └── model_zoo.py      # 模型注册表
├── utils/                 # 工具函数
│   ├── loss.py           # 损失函数
│   ├── metrics.py        # 评估指标
│   ├── saver.py          # 模型保存
│   └── ...
├── data_wenjian/         # NEU-SEG数据集
├── data_magnetic/        # MagneticTile数据集
├── train.py              # 训练脚本
├── val.py               # 验证脚本
├── predict.py           # 推理脚本
└── mypath.py            # 路径配置
```



## 📝 开发指南

### 添加新的改进模块

1. **在`blocks/`目录下创建新模块**
```python
# blocks/MyModule.py
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 实现你的模块
        
    def forward(self, x):
        # 前向传播
        return x
```

2. **在`unet/unet_parts.py`中集成**
```python
from blocks import MyModule

class MyUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        # 使用你的模块
```

3. **在`unet/model_zoo.py`中注册**
```python
MODEL_ZOO = {
    'MyUNet': MyUNet,
    # ... 其他模型
}
```

### 添加新的数据集

1. **在`dataloaders/datasets/`下创建数据集类**
2. **在`dataloaders/__init__.py`中添加数据加载逻辑**
3. **在`mypath.py`中配置数据集路径**

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。


---

<div align="center">



</div>

