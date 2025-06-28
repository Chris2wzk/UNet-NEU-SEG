from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class MyDataset(Dataset):

    NUM_CLASSES = 6

    def __init__(self, args, split="train"):
        self.root = Path.db_root_dir(args.dataset)
        self.split = split
        self.args = args
        self.images = []
        self.categories = []
        
        # 读取数据集文件列表
        _splits_dir = os.path.join(self.root, 'ImageSets', 'Segmentation')
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.target_dir = os.path.join(self.root, 'SegmentationClass')
        
        split_f = os.path.join(_splits_dir, self.split + '.txt')
        if not os.path.exists(split_f):
            raise ValueError(f'Invalid split name: {self.split}')
            
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(self.image_dir, x + ".jpg") for x in file_names]
        self.categories = [os.path.join(self.target_dir, x + ".png") for x in file_names]
        
        assert (len(self.images) == len(self.categories))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_val(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(512),
            tr.RandomHorizontalFlip(),
            tr.RandomRotate(180),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(512),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return composed_transforms(sample)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 200
    args.crop_size = 200

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


