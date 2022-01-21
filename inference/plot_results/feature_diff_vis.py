import os
import cv2
import PIL
import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torchvision.utils as utils

img_0_feature = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/new_81_16011578/features_0.pt')
img_1_feature = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/new_81_16011578/features_1.pt')
img_2_feature = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/new_81_16011578/features_2.pt')
# img_3_feature = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/84_35321310/features_3.pt')
# img_4_feature = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/84_35321310/features_4.pt')

img01_diff_feature = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/84_35321310/features_diff_1.pt')
img12_diff_feature = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/84_35321310/features_diff_2.pt')
# img23_diff_feature = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/84_35321310/features_diff_3.pt')
# img34_diff_feature = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/84_35321310/features_diff_4.pt')
# # # print(img_1_feature[0].shape)
# # plt.imshow(img_1_feature[0].numpy())
# # layers 0
#
# # obtain the overlaid cnn map of image 1
feature = img_2_feature[3][9, ...] - img_1_feature[3][9, ...]
feature[feature<0]=0
image_1 = PIL.Image.open('/image_diff/new_81_16011578/aug_images_cc/image_1/image_1_diff.png')
feat_map_0 = feature.cpu().numpy().transpose(1, 2, 0)[..., 23]
feat_map_0 = cv2.resize(feat_map_0, image_1.size)
cv2.normalize(feat_map_0, feat_map_0, 0, 255, cv2.NORM_MINMAX)

ovl_img_0 = 0.6*np.array(image_1).astype(np.float) + np.stack([feat_map_0, feat_map_0, feat_map_0], axis=2).astype(np.float)
cv2.normalize(ovl_img_0, ovl_img_0, 0, 255, cv2.NORM_MINMAX)
plt.figure()
plt.imshow(ovl_img_0.astype(np.uint8))
io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/new_81_16011578/feature_maps/ovl_image_4.png', ovl_img_0.astype(np.uint8))
# io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/feat_map_0.png', feat_map_0.astype(np.uint8))
#
# # obtain the overlaid cnn map of image 1
# image_1 = PIL.Image.open('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/image_diff_1/image_diff_1.png')
# feat_map_1 = img_1_feature[0][9, ...].cpu().numpy().transpose(1, 2, 0)[..., 23]
# feat_map_1 = cv2.resize(feat_map_1, image_1.size)
# cv2.normalize(feat_map_1, feat_map_1, 0, 255, cv2.NORM_MINMAX)
#
# ovl_img_1 = 0.6*np.array(image_1).astype(np.float) + np.stack([feat_map_1, feat_map_1, feat_map_1], axis=2).astype(np.float)
# cv2.normalize(ovl_img_1, ovl_img_1, 0, 255, cv2.NORM_MINMAX)
# plt.figure()
# plt.imshow(ovl_img_1.astype(np.uint8))
# io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/ovl_image_1.png', ovl_img_1.astype(np.uint8))
# io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/feat_map_1.png', feat_map_1.astype(np.uint8))
#
# # obtain the overlaid cnn map of image 2
# image_2 = PIL.Image.open('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/image_diff_1//image_diff_2.png')
# feat_map_2 = img_2_feature[0][9, ...].cpu().numpy().transpose(1, 2, 0)[..., 12]
# feat_map_2 = cv2.resize(feat_map_2, image_2.size)
# cv2.normalize(feat_map_2, feat_map_2, 0, 255, cv2.NORM_MINMAX)
#
# ovl_img_2 = 0.6*np.array(image_2).astype(np.float) + np.stack([feat_map_2, feat_map_2, feat_map_2], axis=2).astype(np.float)
# cv2.normalize(ovl_img_2, ovl_img_2, 0, 255, cv2.NORM_MINMAX)
# plt.figure()
# plt.imshow(ovl_img_2.astype(np.uint8))
# io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/feat_map_2.png', feat_map_2.astype(np.uint8))
# io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/ovl_image_2.png', ovl_img_2.astype(np.uint8))
#
# # obtain the overlaid cnn map of diff image 12
# diff_image_12 = PIL.Image.open('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/image_diff_1_diff.png')
# feat_map_diff = img12_diff_feature[0][9, ...].cpu().numpy().transpose(1, 2, 0)[..., 23]
# feat_map_diff[feat_map_diff < 0] = 0
# feat_map_diff = cv2.resize(feat_map_diff, diff_image_12.size)
# cv2.normalize(feat_map_diff, feat_map_diff, 0, 255, cv2.NORM_MINMAX)
# ovl_diff_image = 0.6*np.array(diff_image_12).astype(np.float) + np.stack([feat_map_diff, feat_map_diff, feat_map_diff], axis=2).astype(np.float)
# cv2.normalize(ovl_diff_image, ovl_diff_image, 0, 255, cv2.NORM_MINMAX)
# io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/ovl_diff_image.png', ovl_diff_image.astype(np.uint8))
# io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/feat_map_diff.png', feat_map_diff.astype(np.uint8))
# plt.imshow(feat_map.astype(np.uint8))

# grid_map = utils.make_grid(img_1_feature[0][9, ...].unsqueeze(dim=1) - img_0_feature[0][9, ...].unsqueeze(dim=1)).cpu().numpy().transpose(1, 2, 0)
# grid_map = utils.make_grid(img_0_feature[0][9, ...].unsqueeze(dim=1)).cpu().numpy().transpose(1, 2, 0)
# # # print(grid_map.dtype)
# # cv2.normalize(grid_map, grid_map, 0, 255, cv2.NORM_MINMAX)
# plt.figure()
# plt.imshow(grid_map)
# plt.imshow(grid_map, 'gray')
# a = img_2_feature[0][9, ...] - img_1_feature[0][9, ...]
# a[a < 0] = 0
# a = a.sum(dim=0).cpu().numpy()
# cv2.normalize(a, a, 0, 255, cv2.NORM_MINMAX)
# plt.imshow(a)
# substract the feature map
# a2 = img_2_feature[0][9, ...].cpu().numpy().transpose(1, 2, 0)[..., 12]
# a1 = img_1_feature[0][9, ...].cpu().numpy().transpose(1, 2, 0)[..., 23]
# a0 = img_0_feature[0][9, ...].cpu().numpy().transpose(1, 2, 0)[..., 48]
# a = a1 - a0
# a[a < 0] = 0
# cv2.normalize(a, a, 0, 255, cv2.NORM_MINMAX)
# plt.imshow(a.astype(np.uint8))
# io.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/81_16011578/ovl_diff_image0.png', a.astype(np.uint8))
# plt.figure()
# plt.imshow(ovl_diff_image.astype(np.uint8))
# plt.show()
#
feature = (img_2_feature[0][9, ...] - img_1_feature[0][9, ...]).cpu().numpy().transpose(1, 2, 0)
feature_sum = img_1_feature[3][9, ...].cpu().numpy().transpose(1, 2, 0).sum(axis=-1)
feature_sum[feature_sum < 0] = 0
cv2.normalize(feature_sum, feature_sum, 0, 255, cv2.NORM_MINMAX)
feature_sum = cv2.resize(feature_sum, (320, 320))

# img = io.imread('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/new_81_16011578/aug_images_cc/image_1/image_1.png')
# a = PIL.Image.fromarray(feature_sum.astype(np.uint8))
# a = PIL.ImageEnhance.Contrast(a).enhance(3)
# a.show()
# plt.imshow(feature_sum.astype(np.uint8))
# feature_sum = np.array(a)
# ovl_img = 0.6*img.astype(np.float) + np.stack([feature_sum, feature_sum, feature_sum], axis=2).astype(np.float)
# cv2.normalize(ovl_img, ovl_img, 0, 255, cv2.NORM_MINMAX)
# plt.imshow(ovl_img.astype(np.uint8))
plt.show()
# plt.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/new_81_16011578/feature_maps/feature_diff_45/f4.png',
#            feature_sum.astype(np.uint8))
# feature = (img_2_feature[3][9, ...] - img_1_feature[3][9, ...]).cpu().numpy().transpose(1, 2, 0)
# # feature = img_2_feature[0][9, ...].cpu().numpy().transpose(1, 2, 0)
# for i in range(feature.shape[-1]):
#     img = feature[:, :, i]
#     img[img < 0] = 0
#     cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
#     img = cv2.resize(img, (320, 320))
#     plt.imsave('/home/zyi/MedicalAI/Skin_lesion_prognosis/image_diff/new_81_16011578/feature_maps/feature_diff_23/block_4/feat_diff_{}.png'.format(i),
#                img.astype(np.uint8))

