import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from functools import lru_cache


def load_image(filename):
    ext = str(filename).split('.')[-1]
    if ext == 'npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['pt', 'pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


class BasicDataset(Dataset):
    def __init__(self, images_dir, mask_dir, scale=1.0, transform=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.transform = transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [file.stem for file in self.images_dir.glob('*') if not file.name.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values...')
        self.mask_values = self.get_mask_values()
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @lru_cache(maxsize=None)
    def get_mask_values(self):
        """获取所有掩码中的唯一值"""
        unique_values = set()
        for name in self.ids:
            mask_file = list(self.mask_dir.glob(name + '.*'))[0]
            mask = np.asarray(load_image(mask_file))
            if mask.ndim == 2:
                unique_values.update(np.unique(mask))
            elif mask.ndim == 3:
                unique_values.update(np.unique(mask.reshape(-1, mask.shape[-1]), axis=0))
        return sorted(list(unique_values))

    def preprocess(self, img, is_mask):
        """预处理图像和掩码"""
        w, h = img.size
        newW, newH = int(w * self.scale), int(h * self.scale)
        assert newW > 0 and newH > 0, f'Scale {self.scale} is too small, resized images would have no pixel'
        
        if not is_mask:
            img = img.resize((newW, newH), resample=Image.BICUBIC)
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            return img / 255.0
        else:
            img = img.resize((newW, newH), resample=Image.NEAREST)
            return np.asarray(img)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))[0]
        img_file = list(self.images_dir.glob(name + '.*'))[0]

        mask = load_image(mask_file)
        img = load_image(img_file)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        # 应用数据增强
        if self.transform is not None:
            sample = {'image': img, 'mask': mask}
            sample = self.transform(sample)
            return sample
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
