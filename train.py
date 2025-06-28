import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm
import datetime

from mypath import Path
from dataloaders import make_data_loader
from unet.unet_model import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torch.cuda.amp import autocast, GradScaler

def set_seed(seed: int):
    """
    设置随机种子以确保结果可复现。
    如果想更加严格地固定随机性，建议还在 Dataloader 中使用 worker_init_fn。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """
    DataLoader 在多进程下，每个 worker 的 random seed 建议也固定。
    这样能降低不同程间的随机性带来的影响。
    """
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.scaler = GradScaler()

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader with optimized settings
        kwargs = {
            'num_workers': args.workers,
            'pin_memory': True,
            'prefetch_factor': 2,
            'worker_init_fn': worker_init_fn  # 若需固定每个 DataLoader worker 的随机数
        }
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model_class = self._get_model_class(args.model_type)
        model = model_class(n_channels=3, n_classes=4)
        train_params = [{'params': model.parameters(), 'lr': args.lr}]

        # Define Optimizer
        optimizer = torch.optim.SGD(
            train_params,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )

        # Define Criterion
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        self.model, self.optimizer = model, optimizer

        # Define Evaluator & LR Scheduler
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # 若不想加载任何模型，下面这块儿可直接注释或删除
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        # 日志文件路径
        log_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        os.makedirs('runs', exist_ok=True)
        self.log_path = os.path.join('runs', f'train_log_{log_time}.txt')
        with open(self.log_path, 'w') as f:
            f.write('')  # 清空内容，准备写 summary

    def _get_model_class(self, model_type):
        """根据 model_type 参数获取对应的模型类"""
        if model_type in globals():
            model_class = globals()[model_type]
        else:
            raise ValueError(f"未找到指定的模型类型: {model_type}")
        return model_class

    def training(self, epoch):
        self.model.train()
        train_loss = 0.0
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            with autocast():
                output = self.model(image)
                loss = self.criterion(output, target)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
            if i % 10 == 0:
                torch.cuda.empty_cache()
        avg_train_loss = train_loss / num_img_tr
        self.writer.add_scalar('train/total_loss_epoch', avg_train_loss, epoch)
        summary_str = f"=>Epoches {epoch}, learning rate = {self.optimizer.param_groups[0]['lr']:.4f}, previous best = {self.best_pred:.4f}\n"
        summary_str += f"[Epoch: {epoch}, numImages: {i * self.args.batch_size + image.data.shape[0]:5d}]\n"
        summary_str += f"Loss: {avg_train_loss:.3f}\n"
        print(summary_str, end='')
        with open(self.log_path, 'a') as f:
            f.write(summary_str)
        if self.args.no_val:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                "model_name": self.model.__class__.__name__,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        return avg_train_loss

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)
        avg_val_loss = test_loss / len(self.val_loader)
        Acc = self.evaluator.Pixel_Accuracy()
        if isinstance(Acc, tuple):
            Acc = Acc[0]
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', avg_val_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        summary_str = f"Validation:\n"
        summary_str += f"[Epoch: {epoch}, numImages: {i * self.args.batch_size + image.data.shape[0]:5d}]\n"
        summary_str += f"Acc:{Acc:.4f}, Acc_class:{Acc_class:.4f}, mIoU:{mIoU:.4f}, fwIoU:{FWIoU:.4f}\n"
        summary_str += f"Loss: {avg_val_loss:.3f}\n"
        print(summary_str, end='')
        with open(self.log_path, 'a') as f:
            f.write(summary_str)
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        return avg_val_loss

def main():
    parser = argparse.ArgumentParser(description="PyTorch Unet Training")
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'])
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--test-batch-size', type=int, default=None)
    parser.add_argument('--use-balanced-weights', action='store_true', default=False)
    # optimizer params
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos','elr'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--nesterov', action='store_true', default=False)
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42,
                        help='可使用此值来固定随机种子')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-val', action='store_true', default=False)
    parser.add_argument('--model_type', type=str, default='UNet',
                        help='选择模型类型，对应模型类名，例如 Unet, UnetAttention1 等')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # 设定默认 epochs, batch_size, lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 8

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.05,
        }
        args.lr = lrs[args.dataset.lower()] / (2 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        time_str = datetime.datetime.now().strftime('%m%d_%H%M')
        args.checkname = f"{args.model_type}_{time_str}"
    else:
        time_str = datetime.datetime.now().strftime('%m%d_%H%M')
        args.checkname = f"{args.checkname}_{time_str}"
    print(args)

    # 在使用 Trainer 前先固定随机种子
    set_seed(args.seed)

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.avg_train_loss = trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
    main()
