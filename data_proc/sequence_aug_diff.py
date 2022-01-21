import numpy as np
import torch
import random
import PIL
import cv2
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
from torchvision.transforms import RandomResizedCrop, ColorJitter, Lambda, Compose
import torchvision.transforms.functional as F
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
from skimage.transform import warp
from skimage.transform import EuclideanTransform

class Affine:
    def __init__(self, augs):
        self.affine = iaa.Affine(
            rotate=(-augs['rotation'], augs['rotation']),
            shear=(-augs['shear'], augs['shear']),
            scale=({'x': augs['scale'][0], 'y': augs['scale'][1]}),
            mode='symmetric')

    def random_affine(self, images_list):
        if len(images_list) > 1:
            try:
                images_list = list(map(np.array, images_list))
                images = np.concatenate(images_list, axis=2)
            except ValueError:
                print('images in the sequence have different shape!!!')
        else:
            images = np.array(images_list[0])

        assert images.shape[-1] % 3 == 0
        images = self.affine.augment_image(images)

        images_list = []

        if images.shape[-1] == 3:
            images_list.append(PIL.Image.fromarray(images))
        else:
            for i in range(images.shape[-1]//3):
                images_list.append(PIL.Image.fromarray(images[:, :, i*3:(i+1)*3]))

        return images_list

    def __call__(self, images_list):
        assert isinstance(images_list, list)

        if random.random() < 0.5:
            images_list = self.random_affine(images_list)

        return images_list


class RandomCrop(RandomResizedCrop):
    def __init__(self, size, scale, ratio):
        super(RandomCrop, self).__init__(size, scale, ratio)

    def compute_common_size(self, images_list):
        aspect_ratio = []
        for i in range(len(images_list)):
            aspect_ratio.append(images_list[i].size[0]/images_list[i].size[1])

        _, index, count = np.unique(aspect_ratio, return_index=True, return_counts=True)
        index = index[np.where(count == np.max(count))]

        if len(index) > 1:
            index = index[np.random.randint(len(index))]

        try:
            size = images_list[index].size
        except TypeError:
            size = images_list[index[0]].size

        return size

    def uniform_img_sequence_size(self, images_list):
        assert isinstance(images_list, list)

        if len(images_list) > 1:
            size = self.compute_common_size(images_list)
            images_list = [img.resize(size) for img in images_list]

        return images_list

    def __call__(self, images_list):
        assert isinstance(images_list, list)
        images_list = self.uniform_img_sequence_size(images_list)

        x, y, h, w = self.get_params(images_list[0], self.scale, self.ratio)

        for i in range(len(images_list)):
            images_list[i] = F.resized_crop(images_list[i], x, y, h, w, self.size, self.interpolation)

        return images_list


class RamdomFlip:
    def __init__(self, p=0.5, test_mode=False):
        self.p = p
        self.test_mode = test_mode

    def horizontal_flip(self, images_list):
        images_list = list(map(F.hflip, images_list))
        return images_list

    def vertical_flip(self, images_list):
        images_list = list(map(F.vflip, images_list))
        return images_list

    def mixed_flip(self, images_list):
        images_list = self.vertical_flip(self.horizontal_flip(images_list))
        return images_list

    def __call__(self, images_list):
        assert isinstance(images_list, list)

        if not self.test_mode:
            if random.random() < self.p:
                images_list = self.horizontal_flip(images_list)

            if random.random() > self.p:
                images_list = self.vertical_flip(images_list)

            return images_list

        else:
            new_images_list = [images_list, self.vertical_flip(images_list), self.horizontal_flip(images_list),
                               self.mixed_flip(images_list)]

            return new_images_list


class RandomColorJitter(ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue, color_recalibration=False, padding_mode='normal'):
        super(RandomColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.color_recalib = color_recalibration
        self.padding_mode = padding_mode

    def get_params(self, brightness, contrast, saturation, hue):
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def color_constancy(self, image):
        image = np.array(to_pil(cca.grey_world(from_pil(image))))
        return image

    def compute_diff_img(self, img1, img2):
        if np.all(img1 == img2):
            if self.padding_mode == 'normal':
                delta_x = np.random.random_integers(-10, 10)
                delta_y = np.random.random_integers(-10, 10)
                trans_matrix = np.array([[1, .0, delta_x],
                                         [.0, 1, delta_y],
                                         [0., 0., 1.]])

                img1 = warp(img1, EuclideanTransform(matrix=trans_matrix), mode='reflect')
                img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                diff_img = img2.astype(np.float) - img1.astype(np.float)
                image = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

            elif self.padding_mode == 'blank':
                diff_img = img2.astype(np.float) - img1.astype(np.float)
                image = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
            elif self.padding_mode == 'negative':
                diff_img = 0. - img1.astype(np.float)
                image = cv2.normalize(diff_img, None, 60, 255, cv2.NORM_MINMAX)
            else:
                raise ValueError
        else:
            diff_img = img2.astype(np.float) - img1.astype(np.float)
            image = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

        return to_pil(image)

    def __call__(self, images_list):
        assert isinstance(images_list, list)
        color_transform = self.get_params(self.brightness, self.contrast,
                                          self.saturation, self.hue)
        # TODO: color constancy?
        if self.color_recalib:
            diff_images_list = list(map(self.color_constancy, images_list))

            diff_images_list = [self.compute_diff_img(im1, im2)
                                for im1, im2 in zip(diff_images_list[:-1], diff_images_list[1:])]

            # if add image difference between the first and the last image
            # diff_images_list.append(self.color_constancy(images_list[-1]).astype(np.float) - self.color_constancy(images_list[0]).astype(np.float))
            # diff_images_list = list(map(self.norm_to_pil, diff_images_list))
            # add color augmentation for difference image
            diff_images_list = list(map(color_transform, diff_images_list))
            # add color augmentation for RGB image
            images_list = list(map(color_transform, images_list))
        else:
            images_list = list(map(color_transform, images_list))
            diff_images_list = [self.compute_diff_img(im1, im2)
                                for im1, im2 in zip(images_list[:-1], images_list[1:])]

        # print(type(diff_images_list[0]))
        # imt = diff_images_list[0]
        # cv2.normalize(imt, imt, 0, 255, cv2.NORM_MINMAX)
        # plt.imshow(imt.astype(np.uint8))
        # plt.show(block=1)
        # images_list = list(map(self.random_color_trans, images_list))
        return {'images': images_list, 'diff_images': diff_images_list}


class NormalizeTensor:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def normalize(self, image):
        image = F.to_tensor(image)
        # if isinstance(image, np.ndarray):
        #     imt = image.transpose(1, 0).transpose(2, 1).numpy()
        #     cv2.normalize(imt, imt, 0, 255, cv2.NORM_MINMAX)
        #     plt.imshow(imt.astype(np.uint8))
        #     plt.show(block=0)

        return F.normalize(image, self.mean, self.std)

    def __call__(self, inputs_list):
        """
        inputs_list: {'images': xxx, 'diff_images': xxx}
        """
        assert isinstance(inputs_list, list) or isinstance(inputs_list, dict)
        if isinstance(inputs_list, dict):
            images_list = list(map(self.normalize, inputs_list['images']))  # To Tensor
            images_list = torch.stack(images_list, dim=0)

            diff_images_list = list(map(self.normalize, inputs_list['diff_images']))
            diff_images_list = torch.stack(diff_images_list, dim=0)

            return {'images': images_list, 'diff_images': diff_images_list}
        else:
            images_list = list(map(self.normalize, inputs_list))  # To Tensor
            images_list = torch.stack(images_list, dim=0)
        return images_list


class TestCropNorm:
    def __init__(self, size, mean, std, color_augs=None, color_calibration=True, crop_aug='ten_crops', padding_mode='negative'):
        self.size = size
        self.crop_aug = crop_aug
        self.color_augs = color_augs
        self.random_flip = RamdomFlip(test_mode=True)
        self.normalize_tensor = NormalizeTensor(mean, std)
        self.color_recalib = color_calibration
        self.padding_mode = padding_mode
        if color_augs is not None:
            self.color_trans = RandomColorJitter(color_augs['brightness'], color_augs['contrast'],
                                                 color_augs['saturation'], color_augs['hue'])

    def compute_common_size(self, images_list):
        aspect_ratio = []
        for i in range(len(images_list)):
            aspect_ratio.append(images_list[i].size[0] / images_list[i].size[1])

        _, index, count = np.unique(aspect_ratio, return_index=True, return_counts=True)
        index = index[np.where(count == np.max(count))]

        if len(index) > 1:
            index = index[np.random.randint(len(index))]

        try:
            size = images_list[index].size
        except TypeError:
            size = images_list[index[0]].size

        return size

    def uniform_img_sequence_size(self, images_list):
        assert isinstance(images_list, list)

        if len(images_list) > 1:
            size = self.compute_common_size(images_list)
            images_list = [img.resize(size) for img in images_list]   # this is correct

        return images_list

    def center_crop(self, image):
        h, w = image.size
        if h > w:
            image = image.resize((int(self.size * h/w), self.size))
        else:
            image = image.resize((self.size, int(self.size * w/h)))

        return F.center_crop(image, (self.size, self.size))

    def ten_resized_crop(self, image):
        h, w = image.size
        if h > w:
            image = image.resize((int(self.size * h/w), self.size))
        else:
            image = image.resize((self.size, int(self.size * w/h)))

        images_list = F.ten_crop(image, (self.size, self.size), vertical_flip=False)   # Crops & Horizontal Flips
        # images_list = list(images_list) + self.random_flip.vertical_flip(images_list)  # add vertical flips
        # image = image.thumbnail(size, PIL.Image.ANTIALIAS)   # resize image with original aspect ration
        return images_list   # list of PIL images

    def test_flip_aug(self, images_list):
        images_list = list(map(self.center_crop, images_list))
        images_list = self.random_flip(images_list)   # comment here for only use center crop without flip

        return images_list

    def color_constancy(self, image):
        image = np.array(to_pil(cca.grey_world(from_pil(image))))
        return image

    def norm_to_pil(self, image):
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = to_pil(image)
        return image

    def compute_diff_img(self, img1, img2):
        if np.all(img1 == img2):
            if self.padding_mode == 'normal':
                # transform img1 with small pixel translation
                delta_x = np.random.random_integers(-10, 10)
                delta_y = np.random.random_integers(-10, 10)
                trans_matrix = np.array([[1, .0, delta_x],
                                         [.0, 1, delta_y],
                                         [0., 0., 1.]])

                img1 = warp(img1, EuclideanTransform(matrix=trans_matrix), mode='reflect')
                img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # compute the diff images
                diff_img = img2.astype(np.float) - img1.astype(np.float)
                image = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
            elif self.padding_mode == 'blank':
                diff_img = img2.astype(np.float) - img1.astype(np.float)
                image = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
            elif self.padding_mode == 'negative':
                diff_img = 0. - img1.astype(np.float)
                image = cv2.normalize(diff_img, None, 60, 255, cv2.NORM_MINMAX)
            else:
                raise ValueError
        else:
            diff_img = img2.astype(np.float) - img1.astype(np.float)
            image = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
        return to_pil(image)

    def extract_diff_image(self, images_list):
        # TODO: color constancy?
        if self.color_recalib:
            diff_images_list = list(map(self.color_constancy, images_list))

            diff_images_list = [self.compute_diff_img(im1, im2)
                                for im1, im2 in zip(diff_images_list[:-1], diff_images_list[1:])]

            # diff_images_list = [self.color_constancy(im2).astype(np.float) - self.color_constancy(im1).astype(np.float)
            #                     for im1, im2 in zip(images_list[:-1], images_list[1:])]

            # if add the difference image between first and the last image
            # diff_images_list.append(self.color_constancy(images_list[-1]).astype(np.float) - self.color_constancy(images_list[0]).astype(np.float))
        else:
            diff_images_list = [self.compute_diff_img(im1, im2) for im1, im2 in zip(images_list[:-1], images_list[1:])]
            # diff_images_list.append(self.color_constancy(images_list[-1]).astype(np.float) - self.color_constancy(images_list[0]).astype(np.float))

        return diff_images_list

    def __call__(self, images_list):
        assert isinstance(images_list, list)
        images_list = self.uniform_img_sequence_size(images_list)   # resize

        if self.crop_aug == 'ten_crops':
            images_crops_lsit = list(map(self.ten_resized_crop, images_list))  # [[list], [list]]
            # transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
            # reorganize the image crops list, so each sublist is a crop sequence images
            new_images_crops_list = []
            for i in range(len(images_crops_lsit[0])):
                new_images_crops_list.append(list(map(lambda img_list: img_list[i], images_crops_lsit)))

        elif self.crop_aug == 'center_crops':
            # [original_lis, v_flip_list, h_flip_list, hv_flip_list]
            new_images_crops_list = self.test_flip_aug(images_list)

        else:
            raise ValueError

        diff_images_list = list(map(self.extract_diff_image, new_images_crops_list))  # [[t1, t2, t3], [...], ...]
        # if self.color_recalib:
        #     diff_images_list = list(map(self.extract_diff_image, new_images_crops_list))
        #     # if self.color_augs is not None:
        #     #     new_images_crops_list = list(map(self.color_trans, new_images_crops_list))
        # else:
        #     # if self.color_augs is not None:
        #     #     new_images_crops_list = list(map(self.color_trans, new_images_crops_list))
        #     diff_images_list = list(map(self.extract_diff_image, new_images_crops_list))

        # new_images_crops_list = list(map(self.random_flip, new_images_crops_list))

        normalized_crops_list = list(map(self.normalize_tensor, new_images_crops_list))
        normalized_diff_crops_list = list(map(self.normalize_tensor, diff_images_list))  # [T*C*H*W, T*C*H*W...]

        images_list = torch.stack(normalized_crops_list, dim=0)
        diff_images_list = torch.stack(normalized_diff_crops_list, dim=0)
        return {'images': images_list, 'diff_images': diff_images_list}      # N_crops x T x C x H x W


class Augmentations_diff:
    def __init__(self, augs, test_mode=False, color_recalibration=True, test_aug='ten_crops', padding_mode='negative'):
        transform_list = []
        self.test_crop = test_aug
        self.mean = augs['normalization']['mean']
        self.std = augs['normalization']['std']
        self.color_constancy = color_recalibration
        self.padding_mode = padding_mode
        if not test_mode:
            # random resize and crop
            self.random_resize_crop = RandomCrop(size=augs['size'], scale=augs['scale'], ratio=augs['ratio'])
            transform_list.append(self.random_resize_crop)

            # random affine
            if augs['affine'] is not None:
                self.random_affine = Affine(augs['affine'])
                transform_list.append(self.random_affine)

            # random flip
            if augs['flip'] is not None:
                self.random_flip = RamdomFlip()
                transform_list.append(self.random_flip)

            # random color trans
            if augs['color_trans'] is not None:
                self.random_color_trans = RandomColorJitter(augs['color_trans']['brightness'],
                                                            augs['color_trans']['contrast'],
                                                            augs['color_trans']['saturation'],
                                                            augs['color_trans']['hue'],
                                                            self.color_constancy,
                                                            self.padding_mode)

                transform_list.append(self.random_color_trans)

            # normalize to tensor
            self.normalize = NormalizeTensor(mean=augs['normalization']['mean'], std=augs['normalization']['std'])
            transform_list.append(self.normalize)

            self.transform = transforms.Compose(transform_list)  # T x C x H x W

        else:
            # resize + ten crop + color_trans + normalize
            self.transform = transforms.Compose([TestCropNorm(augs['size'], mean=augs['normalization']['mean'],
                                                              std=augs['normalization']['std'],
                                                              color_augs=augs['color_trans'],
                                                              crop_aug=self.test_crop,
                                                              padding_mode=self.padding_mode)]
                                                )  # N_crops x T x C x H x W

