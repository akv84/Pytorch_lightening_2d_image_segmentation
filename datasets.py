
import logging
from os.path import splitext
import os
import numpy as np
from PIL import Image
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from monai.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, pad_list_data_collate 
from monai.transforms import (
    Activations,
    AddChanneld,
    AsChannelFirstd,
    AsChannelLastd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
    EnsureChannelFirstd,
    ToTensord,
)
Image.LOAD_TRUNCATED_IMAGES = True


def get_training_augmentation(height=512, width=512):
    train_transform = [
        
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
            ],
        p=0.5,
        ),
        A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
            ],
        p=0.5,
        ),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
        p=0.5,
        ),
        A.OneOf(
            [
                A.RandomBrightness(p=1),
                A.RandomContrast(p=1),
            ],
        p=0.5,
        ),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(height=512, width=512):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(height, width),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)


def get_post_process_augmentation(n_classes=3):
    post_trans = Compose(
        [
            monai.transforms.EnsureType(), 
            monai.transforms.AsDiscrete(argmax=True, to_onehot=n_classes, threshold=0.5),
            monai.transforms.KeepLargestConnectedComponent(applied_labels=np.arange(n_classes), is_onehot=True, connectivity=2)
        ]
    )    
    return post_trans


class SegDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, resize: tuple = (512, 512), mask_suffix: str = '', transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.resize = resize
        self.mask_suffix = mask_suffix
        self.transform = transform

        self.ids = [f[len(self.mask_suffix):] for f in os.listdir(self.masks_dir)] # if not f.stratwith('.')]
        self.ids = list(set(os.listdir(self.images_dir)).intersection(self.ids))
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, resize, is_mask):
        w, h = pil_img.size
        new_w, new_h = resize[::-1]
        assert new_w > 0 and new_h > 0, 'resize is too small, resized images would have no pixel'
        pil_img = pil_img.resize((new_w, new_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]

        if is_mask and img_ndarray.ndim == 3:
            img_ndarray = img_ndarray[:,:,0]

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = os.path.join(self.images_dir,name)
        mask_file = os.path.join(self.masks_dir, self.mask_suffix+name)

        mask = self.load(mask_file)
        img = self.load(img_file)

        img = self.preprocess(img, self.resize, is_mask=False)
        mask = self.preprocess(mask, self.resize, is_mask=True)

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
            
            return img/255, mask
        else:
            img = torch.as_tensor(img.copy()).float().contiguous()
            mask = torch.as_tensor(mask.copy()).long().contiguous()
            img = img.permute(2,0,1)
            
            return img/255, mask


class CarvanaDataset(SegDataset):
    def __init__(self, images_dir, masks_dir, resize=(360, 480), transform=None):
        super().__init__(images_dir, masks_dir, resize, mask_suffix='_mask', transform=None)


class SegDataModule(pl.LightningDataModule):
    '''
    Data Module
    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    '''

    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2, num_workers=4):
        super().__init__()
        self.batch_size = batch_size  #hparams['batch_size']
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pad_list_data_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=pad_list_data_collate)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=pad_list_data_collate)
        
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=pad_list_data_collate)
        

        