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
from data_proc import padding_img_seq
from data_proc.sequence_aug import Augmentations
from data_proc.sequence_aug_diff import Augmentations_diff


class Skin_Dataset(Dataset):
    def __init__(self, data_root, case_list, seq_length, transform, data_modality='MAC', is_train=True, test_mode=False):
        super(Skin_Dataset, self).__init__()

        self.n_class = 1
        self.data_root = data_root
        self.case_list = case_list
        self.length = seq_length
        self.data_type = data_modality
        self.transform = transform
        self.is_train = is_train     # training data and validation data have different image sampling method
        self.test_mode = test_mode

    def __len__(self):
        return len(self.case_list)

    def sampling_images(self, case_dir_list, test_mode=False):
        # sampling image sequence with specific length for current case folder
        images_list = []
        padding_index = []

        if self.is_train:
            end_index = 1 if len(case_dir_list) <= self.length \
                else np.random.randint(1, (len(case_dir_list) - self.length) + 2)
        else:
            end_index = 1

        for l in range(self.length):
            try:
                img = PIL.Image.open(case_dir_list[-(l + end_index)])
                padding_index.insert(0, 1.)    # if not padding
            except:
                img = PIL.Image.open(case_dir_list[0])
                padding_index.insert(0, 0.)

            images_list.insert(0, img)

        padding_index = torch.tensor(padding_index)
        img_sequence = self.transform(images_list)   # T x C x H x W

        # for only training with difference images, if image length less than required input length, then padding
        # difference images sequence with first images; if image length == 1, then the difference image will be normal
        # spatial image
        # img_sequence = padding_img_seq(padding_index, img_sequence)

        if self.test_mode:
            padding_index = padding_index.repeat(10, 1)

        return img_sequence, padding_index

    def load_sequence(self, folder):
        img_seq = OrderedDict()
        # former ---> latest
        mac_dir_list = sorted(glob(os.path.join(self.data_root, folder, '*MAC*')),
                              key=lambda x: int(x.split('_')[-1].split('.')[0][:8]))  # clinical images

        mic_dir_list = sorted(glob(os.path.join(self.data_root, folder, '*MIC*')),
                              key=lambda x: int(x.split('_')[-1].split('.')[0][:8]))  # dermoscopic images

        # assert len(mac_dir_list) > 0 and len(mic_dir_list) > 0, 'images list should large than 0 !!!'
        # assert len(mac_dir_list) == len(mic_dir_list), 'two kinds of image were unpaired !!!'
        assert len(mic_dir_list) > 0, 'images list should large than 0 !!!'
        if self.data_type == 'MAC' or self.data_type == 'MIX':
            mac_imgs, padding_index = self.sampling_images(mac_dir_list)
            img_seq['MAC'] = mac_imgs
            img_seq['p_index'] = padding_index
        if self.data_type == 'MIC' or self.data_type == 'MIX':
            mic_imgs, padding_index = self.sampling_images(mic_dir_list)
            img_seq['MIC'] = mic_imgs
            img_seq['p_index'] = padding_index
        return img_seq

    def __getitem__(self, idx):
        # select case
        folder = self.case_list[idx]      # e.g. 'SMI_Benign/14950521'
        # # load data
        print('we are load images from ', folder, 'images: {}'.format(len(glob(os.path.join(self.data_root, folder, '*MIC*')))))
        img_seq = self.load_sequence(folder)
        label = 0 if 'Benign' in folder else 1

        # for key in img_seq.keys():
        #     print(img_seq[key].shape)
        label = torch.tensor(label)   # dtype = torch.int64

        return {'image': img_seq, 'target': label}


if __name__ == '__main__':
    data_root = '/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned'

    with open('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/data_setting/data_split.pkl', 'rb') as f:
        case_list = pkl.load(f)

    train_list = case_list[0]['train']
    val_list = case_list[0]['val']

    seq_length = 4
    aug_parameters = OrderedDict({'affine': None,
                                  'flip': True,
                                  'color_trans': {'brightness': (0.8, 1.2),
                                                  'contrast': (0.8, 1.2),
                                                  'saturation': (0.8, 1.2),
                                                  'hue': (-0.03, 0.03)},
                                  'normalization': {'mean': (0.485, 0.456, 0.406),
                                                    'std': (0.229, 0.224, 0.225)},
                                  'size': 320,
                                  'scale': (0.8, 1.2),
                                  'ratio': (0.8, 1.2)
                                  }
                                 )

    augmentor = Augmentations_diff(aug_parameters, test_mode=True, padding_mode='negative')
    sample_dataset = Skin_Dataset(data_root, train_list, seq_length,  augmentor.transform, data_modality='MIC', is_train=True,
                                  test_mode=True)

    idx = 68  # np.random.randint(0, len(case_list))

    fig, axes = plt.subplots(1, seq_length-1)
    sequence = sample_dataset[idx]['image']['MIC']
    images = sequence['images']
    diff_images = sequence['diff_images']
    print(diff_images.shape)
    # print(sequence.shape)
    for i in range(seq_length-1):
        img = diff_images[9, i, ...].numpy().transpose(1, 2, 0)
        print('max img: ', np.max(img), 'min img: ', np.min(img))

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        axes[i].imshow(img.astype(np.uint8))
        # plt.figure()
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.hist(gray.ravel(),256,[0,256])
    #     io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/new_81_16011578/aug_images_cc/image_cc_{}.png'.format(i), img.astype(np.uint8))
    # # img = PIL.Image.fromarray(sequence[0, ...].transpose(0, 1).transpose(1, 2).numpy().astype(np.uint8))
    # # img.show()
    # # print(train_list[idx])
    plt.show()
