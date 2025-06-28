#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
说明：
这是一个示例脚本（基于之前的 val.py 做了修改），
在测试评估阶段额外保存网络预测的彩色 PNG 文件，并将各个类别(0,1,2,3)分别用不同颜色进行区分。

其中：
- 0 表示背景(不高亮、或设为全黑)
- 1,2,3 分别表示三种不同类别的缺陷；
1 表示缺陷类型1 “夹渣(inclusion)”。对应的颜色为红色， (0, 0, 255)。
2 表示缺陷类型2 “斑块(patches)”。对应的颜色为蓝色， (255, 0, 0)。
3 表示缺陷类型3 “划痕(scratches)”。对应的颜色为绿色，(0, 255, 0)。

注：本脚本仅做示例，你需要根据自己项目的路径、模型加载逻辑、类别数等做适当调整。
"""

import argparse
import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import time
from datetime import datetime

# 根据你的项目结构修改 import
from dataloaders import make_data_loader
from utils.metrics import Evaluator
from utils.dice_score import dice_coeff, multiclass_dice_coeff
from unet.model_zoo import MODEL_ZOO  # 确保这个文件包含所有模型类


@torch.inference_mode()
def test_evaluate_and_save_png(
    net, 
    dataloader, 
    device, 
    amp, 
    nclasses, 
    save_dir='./results_vis', 
    test_txt=None
):
    """
    使用测试集进行评估并将预测结果以彩色PNG形式保存，并在推理时记录日志信息到 log 文件中：
    每张图片的文件名、生成时间戳、分割缺陷种类、推理用时 等。
    """
    logging.info("[INFO] 开始 test_evaluate_and_save_png")
    logging.info(f"test_evaluate_and_save_png 接收到的 test_txt 参数: {test_txt}")
    net.eval()
    num_test_batches = len(dataloader)
    dice_score = 0
    evaluator = Evaluator(nclasses)
    total_predictions = 0
    class_counts = np.zeros(nclasses)

    # 读取 test.txt，将其所有行(去空行)放进列表 lines
    if test_txt is not None:
        with open(test_txt, 'r') as fr:
            lines = [x.strip() for x in fr if x.strip()]
    else:
        raise ValueError("必须指定 test_txt 路径")

    # 确保输出目录
    os.makedirs(save_dir, exist_ok=True)

    # 额外准备一个 log 文件，用于记录推理信息
    log_file_path = os.path.join(save_dir, "inference_log.txt")

    # 类别->BGR
    color_map = {
        0: (0, 0, 0),     
        1: (0, 0, 255),   # 红色：夹渣inclusion
        2: (255, 0, 0),   # 蓝色：斑块patches
        3: (0, 255, 0),   # 绿色：划痕scratches
    }

    # 全局计数器，指示处理到第几张
    global_index = 0
    # 预热阶段 (消除首次推理延迟)
    if device.type == 'cuda':
        if hasattr(dataloader.dataset, 'crop_size'):
            warmup_shape = dataloader.dataset.crop_size
        else:
            warmup_shape = (512, 512)  # 默认尺寸 512x512
        warmup_tensor = torch.randn(dataloader.batch_size, 3, *warmup_shape).to(device)
        for _ in range(3):
            _ = net(warmup_tensor)
        torch.cuda.synchronize()

    cast_device = device.type if device.type != 'mps' else 'cpu'
    total_infer_time = 0.0  # 新增总耗时统计
    
    with open(log_file_path, "a", encoding="utf-8") as f_log:
        with torch.autocast(cast_device, enabled=amp):
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_test_batches, desc='测试中', unit='batch', leave=False)):
                images, masks_true = batch['image'], batch['label']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks_true = masks_true.to(device=device, dtype=torch.long)

                # 精确计时逻辑
                if device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_event.record()
                else:
                    infer_start_time = time.perf_counter()

                # 推理
                masks_pred = net(images)

                # 结束计时
                if device.type == 'cuda':
                    end_event.record()
                    torch.cuda.synchronize()
                    batch_infer_time = start_event.elapsed_time(end_event) / 1000  # 毫秒转秒
                else:
                    batch_infer_time = time.perf_counter() - infer_start_time
                
                total_infer_time += batch_infer_time  # 累计总耗时
                batch_size = images.size(0)
                avg_infer_time = batch_infer_time / batch_size

                # 后续处理保持不变
                preds_argmax = masks_pred.argmax(dim=1)


                # Dice/混淆矩阵等统计
                if nclasses == 1:
                    masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
                    dice_score += dice_coeff(masks_pred_bin, masks_true, reduce_batch_first=False)
                else:
                    masks_true_1hot = F.one_hot(masks_true, nclasses).permute(0,3,1,2).float()
                    preds_1hot = F.one_hot(preds_argmax, nclasses).permute(0,3,1,2).float()

                    dice_score += multiclass_dice_coeff(
                        preds_1hot[:, 1:],         # 去掉背景
                        masks_true_1hot[:, 1:],    # 去掉背景
                        reduce_batch_first=False
                    )
                    pred_np = preds_argmax.cpu().numpy()
                    target_np = masks_true.cpu().numpy()
                    evaluator.add_batch(target_np, pred_np)

                    total_predictions += pred_np.size
                    for c in range(nclasses):
                        class_counts[c] += np.sum(pred_np == c)

                images_np = images.cpu().permute(0,2,3,1).numpy()
                preds_argmax_np = preds_argmax.cpu().numpy()

                for i in range(batch_size):
                    # 从 test.txt 中获取对应行
                    if (global_index + i) < len(lines):
                        test_name = lines[global_index + i]
                    else:
                        test_name = f"out_of_range_{global_index + i}"

                    single_pred = preds_argmax_np[i]
                    h, w = single_pred.shape
                    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                    for cls_id, bgr_val in color_map.items():
                        color_mask[single_pred == cls_id] = bgr_val

                    # 纯色 mask
                    mask_save_path = os.path.join(save_dir, f"{test_name}_mask.png")
                    cv2.imwrite(mask_save_path, color_mask)

                    # 叠加原图
                    img_bgr = (images_np[i] * 255.0).clip(0,255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
                    overlay = cv2.addWeighted(img_bgr, 0.5, color_mask, 0.5, 0)
                    overlay_save_path = os.path.join(save_dir, f"{test_name}_overlay.png")
                    cv2.imwrite(overlay_save_path, overlay)

                    # 统计该图片包含了哪些缺陷类别
                    unique_classes = np.unique(single_pred)
                    defect_names = []
                    for uc in unique_classes:
                        if uc == 1:
                            defect_names.append("夹渣(inclusion)")
                        elif uc == 2:
                            defect_names.append("斑块(patches)")
                        elif uc == 3:
                            defect_names.append("划痕(scratches)")
                    if len(defect_names) == 0:
                        defect_str = "无缺陷"
                    else:
                        defect_str = "、".join(defect_names)

                    # 当前时间戳
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 计算背景像素和缺陷像素的占比
                    total_pixels = h * w
                    bg_pixels = np.sum(single_pred == 0)
                    defect_pixels = total_pixels - bg_pixels
                    bg_ratio = bg_pixels / total_pixels * 100
                    defect_ratio = defect_pixels / total_pixels * 100

                    # 记录到 log 文件，同时加入背景和缺陷像素占比信息
                    log_line = (
                        f"图片名称: {test_name}, "
                        f"时间戳: {now_str}, "
                        f"缺陷类别: {defect_str}, "
                        f"推理用时: {avg_infer_time:.4f} 秒, "
                        f"背景占比: {bg_ratio:.2f}%, "
                        f"缺陷占比: {defect_ratio:.2f}%"
                    )
                    f_log.write(log_line + "\n")

                global_index += batch_size

    # 输出统计
    print("\n整体预测统计:")
    print(f"总像素数: {total_predictions}")
    for i in range(nclasses):
        percentage = (class_counts[i] / total_predictions) * 100 if total_predictions > 0 else 0
        print(f"类别 {i}: {class_counts[i]} 像素 ({percentage:.2f}%)")

    print("\n混淆矩阵:")
    print(evaluator.confusion_matrix)

    print("\n每个类别的真实像素数量:")
    print(evaluator.confusion_matrix.sum(axis=1))

    avg_dice = dice_score / max(num_test_batches, 1)
    Acc_all, Acc_fg = evaluator.Pixel_Accuracy()
    pixel_acc_class = evaluator.Pixel_Accuracy_Class()

    class_ious = np.diag(evaluator.confusion_matrix) / (
        np.sum(evaluator.confusion_matrix, axis=1) +
        np.sum(evaluator.confusion_matrix, axis=0) -
        np.diag(evaluator.confusion_matrix) + 1e-10
    )

    miou = evaluator.Mean_Intersection_over_Union()
    fw_iou = evaluator.Frequency_Weighted_Intersection_over_Union()

    print(f"\n最终测试结果:")
    print(f"平均Dice系数: {avg_dice:.3f}")
    print(f"PA(仅前景): {Acc_fg:.3f}")
    print(f"mPA: {pixel_acc_class:.3f}")
    print(f"mIoU: {miou:.3f}")

    return {
        'dice': float(f"{avg_dice:.4f}"),
        'pixel_acc_all': float(f"{Acc_all:.4f}"),
        'pixel_acc_fg': float(f"{Acc_fg:.4f}"),
        'pixel_acc_class': float(f"{pixel_acc_class:.4f}"),
        'miou': float(f"{miou:.4f}"),
        'fw_iou': float(f"{fw_iou:.4f}"),
        'class_ious': [float(f"{iou:.4f}") for iou in class_ious]
    }


def load_model(checkpoint_path, device, **extra_args):
    """
    通用加载函数:
      1) 自动从checkpoint中获取 model_name
      2) 动态构建该模型类实例
      3) 使用 strict=False 加载 state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get('model_name', 'UNet')
    print(f"加载模型: {model_name} ...")
    print(f"从 {checkpoint_path} 加载权重...")

    if model_name not in MODEL_ZOO:
        raise ValueError(f"未知的模型: {model_name}, 请先在 MODEL_ZOO 中注册。")

    ModelClass = MODEL_ZOO[model_name]
    model = ModelClass(**extra_args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['state_dict'], 
        strict=False
    )
    if missing_keys:
        print(f"[WARN] 加载时有以下参数在checkpoint中缺失: {missing_keys}")
    if unexpected_keys:
        print(f"[WARN] 加载时有以下参数在model中不存在: {unexpected_keys}")

    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="PyTorch Unet Validation + 保存预测彩色结果")
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-size', type=int, default=512)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--test-batch-size', type=int, default=None)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--model_type', type=str, default='UNet')
    parser.add_argument('--test-txt', type=str,
                        default='data_wenjian/ImageSets/Segmentation/test.txt')
    parser.add_argument('--resume', type=str, default='run/pascal/UNet_0627_2003/model_best.pth.tar')
    parser.add_argument('--save-dir', type=str, default='results_unet',
                        help="保存预测可视化结果的文件夹")
    parser.add_argument('--attention', action='store_true', default=False, help="是否启用attention功能（默认关闭）")

    args = parser.parse_args()

    # 设置日志等级为DEBUG
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    logging.info(f"命令行传入的 --test-txt 参数: {args.test_txt}")
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.isfile(args.test_txt):
        raise FileNotFoundError(f"测试集 txt 文件不存在: {args.test_txt}")
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"模型权重路径不存在: {args.resume}")

    print("[DEBUG] 准备加载测试数据...")
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    try:
        train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
        print("[DEBUG] 数据加载器创建成功")
    except Exception as e:
        print(f"[ERROR] 创建数据加载器时出错: {str(e)}")
        raise
        
    if test_loader is None:
        print("[ERROR] test_loader 为 None")
        return

    print("加载模型...")
    model_args = {
        'n_channels': 3,
        'n_classes': 4,  # 这里手动指定4类(含背景)
        'bilinear': False
    }
    if args.attention:
        model_args['attention'] = True

    model, ckpt = load_model(
        checkpoint_path=args.resume,
        device=device,
        **model_args
    )
    model.to(device)

    # 并行 (如果多卡)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    print("开始测试集评估 + 保存预测彩色结果...")
    metrics = test_evaluate_and_save_png(
        net=model,
        dataloader=test_loader,  #这里注意要修改一下  
        device=device,
        amp=True,
        nclasses=4,
        save_dir=args.save_dir,
        test_txt=args.test_txt
    )
    print("评估完成, 指标:", metrics)
    print(f"[INFO] 可视化结果已保存到文件夹: {args.save_dir}")


if __name__ == '__main__':
    main()