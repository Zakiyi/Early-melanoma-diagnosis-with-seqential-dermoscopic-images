import os
import cv2
import PIL
import torch
import skimage.io as io
import numpy as np
from glob import glob
import pickle as pkl
from collections import OrderedDict
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from data_proc.sequence_aug import Augmentations
import torchvision.transforms as transforms


class Skin_Dataset(Dataset):
    def __init__(self, data_root, is_train=True):
        super(Skin_Dataset, self).__init__()

        self.n_class = 1
        self.data_root = data_root
        self.is_train = is_train     # training data and validation data have different image sampling method

        self.samples = glob(os.path.join(data_root, '*', '*.png'))
        # self.samples = glob(os.path.join(data_root, '*', '*.jpg'))
        if self.is_train:
            self.transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2),
                                                                        hue=(-0.05, 0.05)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.508, 0.508, 0.508), (0.139, 0.151, 0.162))
                                                ])

        else:
            self.transform = transforms.Compose([transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.508, 0.508, 0.508), (0.139, 0.151, 0.162))
                                                 ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # select case
        img_dir = self.samples[idx]
        # # load data
        img = PIL.Image.open(img_dir)
        if self.transform:
            img = self.transform(img)

        label = 0 if 'benign' in img_dir else 1

        label = torch.tensor(label)   # dtype = torch.int64

        return {'image': img, 'target': label}


if __name__ == '__main__':
    data_root = '/home/zyi/MedicalAI/kaggle_data'

    sample_dataset = Skin_Dataset('/home/zyi/MedicalAI/HR_Diff_data/diff_dataset/train', is_train=True)
    idx = 2  # np.random.randint(0, len(case_list))
    # test_img = sample_dataset[idx]['image'].transpose(0, 1).transpose(1, 2).numpy()*255
    # cv2.normalize(test_img, test_img, 0, 255, cv2.NORM_MINMAX)
    # io.imshow(test_img.astype(np.uint8))
    # plt.show()
    # fig, axes = plt.subplots(1, seq_length-1)
    # sequence = sample_dataset[idx]['image']['MIC']
    # sequence = sequence['images']
    #
    # for i in range(seq_length-1):
    #     img = sequence[i, ...].numpy().transpose(1, 2, 0)*255
    #     print('max img: ', np.max(img), 'min img: ', np.min(img))
    #     cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    #     axes[i].imshow(img.astype(np.uint8))
    #
    # # img = PIL.Image.fromarray(sequence[0, ...].transpose(0, 1).transpose(1, 2).numpy().astype(np.uint8))
    # # img.show()
    # # print(train_list[idx])
    # plt.show()
    pixel_mean = np.zeros(3)
    pixel_std = np.zeros(3)
    k = 1
    for i_batch, sample_batched in enumerate(sample_dataset):
        image = np.array(sample_batched['image'])
        image = image / 255
        pixels = image.reshape((-1, image.shape[2]))
        for pixel in pixels:
            diff = pixel - pixel_mean
            pixel_mean += diff / k
            pixel_std += diff * (pixel - pixel_mean)
            k += 1
    pixel_std = np.sqrt(pixel_std / (k - 2))
    print(pixel_mean)# [5.50180734]
    print(pixel_std)#[8.27773514]
    # [0.50820405 0.50821917 0.50885629]
    # [0.13906963 0.1510186  0.16291242]
