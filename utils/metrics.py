import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        """
        计算两种PA:
        1. 包含背景的整体PA
        2. 只考虑前景类的PA（以"真实"前景数量为分母）
        """
        # 1. overall PA (含背景)
        Acc_all = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        # 2. foreground PA (不含背景)
        #    分母：所有真实前景像素 = confusion_matrix[1:, :].sum()
        #    分子：正确预测为自己类别的前景像素 = diag(1) + diag(2) + ... + diag(num_class-1)
        ground_truth_fg = self.confusion_matrix[1:, :].sum()  # 所有行从1开始，列全要
        correct_fg = np.diag(self.confusion_matrix)[1:].sum()
        
        if ground_truth_fg > 0:
            Acc_fg = correct_fg / ground_truth_fg
        else:
            Acc_fg = 0.0

        return Acc_all, Acc_fg

    def Pixel_Accuracy_Class(self):
        """
        mPA：跳过背景类 (索引 0)，计算每个前景类别的准确率，再取平均
        """
        # 只取第1类到最后一类（跳过背景类0）
        diag = np.diag(self.confusion_matrix)[1:]
        sums = self.confusion_matrix.sum(axis=1)[1:]
        Acc = diag / sums
        Acc = np.nanmean(Acc)  # 这才是真正的 mPA
        return Acc

    def Mean_Intersection_over_Union_With_Background(self):
        """
        计算包含背景的mIoU
        """
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix) + 1e-10
        )
        return np.nanmean(MIoU)

    def Mean_Intersection_over_Union(self):
        """
        计算不含背景的mIoU (只考虑前景类)
        """
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix) + 1e-10
        )
        # 只取前景类别（跳过背景类索引0）
        return np.nanmean(MIoU[1:])

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Per_Class_Accuracy(self):
        """
        计算每个类别的像素准确率
        返回: 包含每个类别准确率的 numpy 数组
        """
        # 对角线上的值（正确预测）除以每行的和（该类别的总像素数）
        acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        return acc

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def get_scores(self):
        """
        计算并返回所有评估指标
        """
        # PA相关指标
        pixel_acc_all, pixel_acc_fg = self.Pixel_Accuracy()  # 整体PA和前景PA
        mean_pixel_acc = self.Pixel_Accuracy_Class()  # mPA (不含背景)
        
        # IoU相关指标
        miou_with_bg = self.Mean_Intersection_over_Union_With_Background()  # mIoU (含背景)
        miou = self.Mean_Intersection_over_Union()  # mIoU (不含背景)
        fw_iou = self.Frequency_Weighted_Intersection_over_Union()  # FWIoU
        
        # 每个类别的IoU
        class_ious = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix) + 1e-10
        )

        return {
            'PA': pixel_acc_all,
            'PA_fg': pixel_acc_fg,
            'mPA': mean_pixel_acc,
            'mIoU': miou,
            'mIoU_with_bg': miou_with_bg,
            'FWIoU': fw_iou,
            'Class_IoU': class_ious
        }




