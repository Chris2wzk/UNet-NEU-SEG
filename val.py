import argparse
import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 根据你的项目结构修改 import
from dataloaders import make_data_loader
from utils.metrics import Evaluator
from utils.dice_score import dice_coeff, multiclass_dice_coeff
from unet.model_zoo import MODEL_ZOO  # 确保这个文件包含所有模型类


@torch.inference_mode()
def test_evaluate(net, dataloader, device, amp, nclasses):
    """
    使用测试集进行模型评估，计算多个评估指标。
    在原有代码基础上，额外插入少量日志 (DEBUG/INFO) 以便查看网络输出和是否学到特征。
    """
    logging.info("[INFO] 开始 test_evaluate")
    net.eval()
    num_test_batches = len(dataloader)
    dice_score = 0
    evaluator = Evaluator(nclasses)
    total_predictions = 0
    class_counts = np.zeros(nclasses)

    cast_device = device.type if device.type != 'mps' else 'cpu'
    with torch.autocast(cast_device, enabled=amp):
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_test_batches, desc='测试中', unit='batch', leave=False)):
            images, masks_true = batch['image'], batch['label']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks_true = masks_true.to(device=device, dtype=torch.long)

            # 打印输入图像形状、张量范围 (DEBUG)
            logging.debug(f"[DEBUG] Batch={batch_idx}, 输入图像形状={list(images.shape)}, "
                          f"min={float(images.min()):.4f}, max={float(images.max()):.4f}")

            # 前向传播
            masks_pred = net(images)
            if isinstance(masks_pred, (tuple, list)):
                masks_pred = masks_pred[0]

            # 打印网络输出形状、数值范围 (DEBUG)
            logging.debug(f"[DEBUG] Batch={batch_idx}, 输出形状={list(masks_pred.shape)}, "
                          f"min={float(masks_pred.min()):.4f}, max={float(masks_pred.max()):.4f}")

            if nclasses == 1:
                # 二分类
                masks_pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
                dice_score += dice_coeff(masks_pred_bin, masks_true, reduce_batch_first=False)
            else:
                # 多分类
                masks_true_1hot = F.one_hot(masks_true, nclasses).permute(0, 3, 1, 2).float()
                masks_pred_1hot = F.one_hot(masks_pred.argmax(dim=1), nclasses).permute(0, 3, 1, 2).float()

                # 打印各通道信息 (DEBUG)
                for ch_idx in range(masks_pred.shape[1]):
                    ch_data = masks_pred[:, ch_idx, ...]
                    logging.debug(
                        f"[DEBUG] Batch={batch_idx}, 类通道{ch_idx}: "
                        f"mean={float(ch_data.mean()):.4f}, "
                        f"max={float(ch_data.max()):.4f}, "
                        f"min={float(ch_data.min()):.4f}"
                    )

                dice_score += multiclass_dice_coeff(
                    masks_pred_1hot[:, 1:], 
                    masks_true_1hot[:, 1:], 
                    reduce_batch_first=False
                )
                pred = masks_pred.argmax(dim=1).cpu().numpy()
                target = masks_true.cpu().numpy()
                evaluator.add_batch(target, pred)

                total_predictions += pred.size
                for i in range(nclasses):
                    class_counts[i] += np.sum(pred == i)

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
    # print(f"FWIoU: {fw_iou:.4f}")

    # 新添加代码: 打印各类别的像素准确率 (PA)
    per_class_acc = evaluator.Per_Class_Accuracy()
    class_names = {0: "Background", 1: "Inclusion", 2: "Patches", 3: "Scratch"}
    print("\n各类别的像素准确率 (PA):")
    for idx, acc in enumerate(per_class_acc):
        print(f"{class_names.get(idx, f'类别{idx}')}: {acc:.3f}")

    # 新添加代码: 打印各类别的 IoU
    print("\n各类别的 IoU:")
    for idx, iou in enumerate(class_ious):
        print(f"{class_names.get(idx, f'类别{idx}')}: {iou:.3f}")

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

    # 如果模型构造函数不接受 attention 这种额外参数，就从 extra_args 中删除
    if not hasattr(ModelClass, '__init__') or 'attention' not in ModelClass.__init__.__code__.co_varnames:
        extra_args.pop('attention', None)

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
    parser = argparse.ArgumentParser(description="PyTorch Unet Validation")
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
                        default='/data/NEU_SEG/data_wenjian/ImageSets/Segmentation/test.txt')
    parser.add_argument('--resume', type=str, default='run/pascal/exp/UNet_RepLKBlockConv_FastKAN_epoch150_20250207-210814_0207_2108/model_best.pth.tar')

    args = parser.parse_args()

    # 设置日志等级为DEBUG，便于观察调试输出
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.isfile(args.test_txt):
        raise FileNotFoundError(f"测试集 txt 文件不存在: {args.test_txt}")
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"模型权重路径不存在: {args.resume}")

    print("加载测试数据...")
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
    if test_loader is None:
        raise ValueError("test_loader is None! 请检查 make_data_loader 内部逻辑或 txt 文件是否有效。")

    print("加载模型...")
    # 注意: 只在模型实际支持 attention 时再传这类参数
    # 否则多传了参数会导致 TypeError
    model, ckpt = load_model(
        checkpoint_path=args.resume,
        device=device,
        n_channels=3,
        n_classes=4,  # 这里手动指定4类(含背景)，与训练一致
        bilinear=False,
        attention=True
    )
    model.to(device)

    # 并行 (如果多卡)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    print("开始测试集评估...")
    metrics = test_evaluate(model, test_loader, device, amp=True, nclasses=4)
    # print("评估完成, 指标:", metrics)


if __name__ == '__main__':
    main()
