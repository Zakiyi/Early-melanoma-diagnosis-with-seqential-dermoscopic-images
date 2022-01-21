import os
import cv2
import math
from PIL import Image
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
from glob import glob
from skimage.transform import warp, EuclideanTransform, SimilarityTransform


def hair_removal(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # for hair removal
    kernel = cv2.getStructuringElement(1, (17, 17))
    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
    return dst


def compute_image_diff(im1_dir, im2_dir, out_dir=None, color_constancy=True, hair_rm=True):

    im1 = Image.open(im1_dir)
    im2 = Image.open(im2_dir)

    im1 = im1.resize((320, 400))
    if im1.size != im2.size:
        im1 = im1.resize(im2.size)

    # color consistancy
    if color_constancy:
        im1 = to_pil(cca.grey_world(from_pil(im1)))
        im2 = to_pil(cca.grey_world(from_pil(im2)))

    if hair_rm:
        im1 = hair_removal(np.array(im1))
        im2 = hair_removal(np.array(im2))
    else:
        im1 = np.array(im1)
        im2 = np.array(im2)

    if np.all(im1 == im2):
        delta_x = np.random.random_integers(-30, 30)
        delta_y = np.random.random_integers(-30, 30)
        trans_matrix = np.array([[1, .0, delta_x],
                                 [.0, 1, delta_y],
                                 [0., 0., 1.]])

        im1 = warp(im1, EuclideanTransform(matrix=trans_matrix), mode='reflect')
        im1 = cv2.normalize(im1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    image_diff = im2.astype(np.float) - im1.astype(np.float)
    image_diff = cv2.normalize(image_diff, None, 0, 255, cv2.NORM_MINMAX)
    fig, axes = plt.subplots(ncols=1, figsize=(3, 2.4))
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.02, hspace=None)
    plt.axis('off')
    axes.imshow(image_diff.astype(np.uint8))
    # plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/registration/14450763/diff.png', dpi=300)
    plt.show()
    # plt.figure()
    # image_diff = cv2.cvtColor(image_diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # plt.hist(image_diff.ravel(), 256, [0, 256])
    # plt.show()
    #save_dir = os.path.join(out_dir, os.path.basename(im1_dir).split('.')[0])
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    # io.imsave('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/registration/14450763/diffs.png', image_diff.astype(np.uint8))
    #io.imsave(os.path.join(save_dir, os.path.basename(im1_dir)), im1.astype(np.uint8))
    #io.imsave(os.path.join(save_dir, os.path.basename(im2_dir)), im2.astype(np.uint8))
    #io.imsave(os.path.join(save_dir, os.path.basename(im1_dir).split('.')[0] + '_diff.png'), image_diff.astype(np.uint8))

    return image_diff


if __name__ == '__main__':
   im1 = '/home/zyi/MedicalAI/Original skin data/24010622/24010622_MIC_20080912.jpg'
   im2 = '/home/zyi/MedicalAI/Original skin data/24010622/24010622_MIC_20100428.jpg'
   output_dir = '/home/zyi/MedicalAI/HM_10000/images/ISIC_0024608.jpg'
   compute_image_diff(im1, im2, out_dir=output_dir)

   # image_list = glob(os.path.join('/home/zyi/MedicalAI/HR_Serial_Skin_data/SMI_Malignant/111_24551278', '*MIC*'))
   # for im1, im2 in zip(image_list[:-1], image_list[1:]):
   #     compute_image_diff(im1, im2, out_dir=output_dir)

   # case_list = glob(os.path.join('/home/zyi/MedicalAI/HR_Serial_Skin_data_320', '*SMI*', '*'))

   # for case in case_list:
   #     img_list = glob(os.path.join(case, '*MIC*'))
   #     img_list = sorted(img_list, key=lambda x: x.split('/')[-1].split('_')[-1])
   #     out_dir = os.path.join('/home/zyi/MedicalAI/HR_Diff_data/No_Aligned', *case.split('/')[-2:])
   #
   #     if not os.path.exists(out_dir):
   #         os.makedirs(out_dir)
   #
   #     if len(img_list) > 1:
   #         diff_img_list = [compute_image_diff(x1, x2) for x1, x2 in zip(img_list[:-1], img_list[1:])]
   #     else:
   #         print('only 1 image', case)
   #         diff_img_list = [compute_image_diff(img_list[0], img_list[0])]
   #
   #     for l in range(len(diff_img_list)):
   #         io.imsave(os.path.join(out_dir, '{}.png'.format(l)), diff_img_list[l].astype(np.uint8))
   # delta_x = np.random.random_integers(-30, 30)
   # delta_y = np.random.random_integers(-30, 30)
   # trans_matrix = np.array([[1, .0, delta_x],
   #                          [.0, 1, delta_y],
   #                          [0., 0., 1.]])

   # im2 = warp(im1, SimilarityTransform(scale=0.8, rotation=0.14, translation=(1, -2)), mode='reflect')
   # im2 = cv2.normalize(im2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
   #
   # # plt.title('Similarity Transformation: scale: 0.8, rotation: 0.14, translation: (1, -2)')
   # plt.subplot(131)
   # io.imshow(im1)
   # plt.subplot(132)
   # io.imshow(im2)
   # image_diff = im2.astype(np.float) - im1.astype(np.float)
   # image_diff = cv2.normalize(image_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.int)
   # plt.subplot(133)
   # io.imshow(image_diff)
   # plt.tight_layout()
