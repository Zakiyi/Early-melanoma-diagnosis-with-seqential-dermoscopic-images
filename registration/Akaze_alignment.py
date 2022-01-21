import numpy as np
import cv2
import os
import json
from matplotlib import pyplot as plt
import skimage.io as io
from skimage.measure import ransac
from skimage.transform import EuclideanTransform, warp
from colorcorrect.util import from_pil, to_pil
import matplotlib
from _collections import OrderedDict
import colorcorrect.algorithm as cca
matplotlib.style.use('seaborn')


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


def compute_image_diff(im1, im2, color_constancy=True, hair_rm=True):

    if im1.size != im2.size:
        im1 = im1.resize(im2.size)

    if color_constancy:
        im1 = to_pil(cca.grey_world(im1))
        im2 = to_pil(cca.grey_world(im2))

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

    im1 = cv2.normalize(im1, None, 0, 255, cv2.NORM_MINMAX)
    im2 = cv2.normalize(im2, None, 0, 255, cv2.NORM_MINMAX)

    image_diff = im2.astype(np.float) - im1.astype(np.float)
    image_diff = cv2.normalize(image_diff, None, 0, 255, cv2.NORM_MINMAX)

    return image_diff


def convert_points(kp1, kp2, matches):
    """
    convert points and matches in opencv object as numpy array
    so as to fit the input format of skimage.measure.ransac for transformation estimation

    :param kp1: a list of KeyPoint object
    :param kp2: ~
    :param matches: a list of DMatch object
    :return: array of key point coordinates
    """
    kp_1 = [[x.pt[0], x.pt[1]] for x in kp1]
    kp_2 = [[x.pt[0], x.pt[1]] for x in kp2]

    matches = np.array([[m.queryIdx, m.trainIdx] for m in matches])
    src_points = np.array([kp_1[i] for i in matches[:, 0]])  # N * 2
    des_points = np.array([kp_2[i] for i in matches[:, 1]])  # N *2

    return src_points, des_points


def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals

    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def AKAZE_aligment(img1, img2, plot_only_inliers=True, threshold=0.00015):
    """
    :param img1: RGB image in numpy array
    :param img2:
    :param plot_only_inliers:
    :return:
    """

    # Initiate KAZE detector
    detector = cv2.AKAZE_create(descriptor_size=0, threshold=threshold, nOctaves=4)   # 0.0008

    # find the keypoints and compute the descriptors with ORB
    kp1, des1 = detector.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = detector.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    # print(type(des1))
    # img11 = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    # img22 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
    #
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)  # list
    print('des1 ', des1.shape)
    print('des2 ', des2.shape)
    print('kp1 ', kp1[0])
    print('kp2 ', kp2[0])
    print(len(matches))
    matches = sorted(matches, key=lambda x: x.distance)
    print('imgIdx ', matches[0].imgIdx)  # queryIdx, trainIdx
    print('queryIdx ', matches[1].queryIdx)
    print('trainIdx ', matches[1].trainIdx)
    # print(len(matches))

    src_points, des_points = convert_points(kp1, kp2, matches)
    print(src_points.shape)
    print(des_points.shape)
    model, inliers = ransac((src_points, des_points), EuclideanTransform,
                            min_samples=2, residual_threshold=2, max_trials=5000)

    if plot_only_inliers:
        m = []
        for k in range(len(inliers)):
            if inliers[k]:
                m.append(matches[k])

        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, m, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig, axes = plt.subplots(ncols=1, figsize=(8, 4))
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.02, hspace=None)
    axes.imshow(img_matches)
    plt.axis('off')

    plt.show(block=0)

    return model, img_matches


if __name__ == '__main__':
    # data_dir = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854
    img1 = io.imread('/home/zyi/MedicalAI/Original skin data/replaced_benign/14950799')
    img2 = io.imread('/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned/SMI_Malignant/36970572/36970572_MIC_20160722.jpg')

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    model, img_matches = AKAZE_aligment(img2, img1)
    print(model.params)
    print(model.rotation)
    print(list(model.translation))

    img_warp = warp(img2, model.inverse, output_shape=img1.shape, mode='reflect', order=1)
    img_warp = cv2.normalize(img_warp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(img1)
    axes[0].set_axis_off()
    axes[1].imshow(img_warp)
    axes[1].set_axis_off()
    axes[2].imshow(img2)
    axes[2].set_axis_off()
    plt.axis('off')
    plt.tight_layout()

    plt.show()
    #
    # with open(os.path.join(data_dir, 'trans.json'), 'w') as f:
    #     json.dump(OrderedDict({'rotation': model.rotation, 'translation': list(model.translation),
    #                            'threshold': threshold}), f)
    #
    # img_diff_0 = compute_image_diff(img2, img1)
    # img_diff_1 = compute_image_diff(img2, img_warp)
    # #
    # io.imsave(os.path.join(data_dir, 'no_aligned.png'), img_diff_0.astype(np.uint8))
    # io.imsave(os.path.join(data_dir, 'aligned.png'), img_diff_1.astype(np.uint8))
    io.imsave('/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned/SMI_Malignant/36970572warp.png', img_warp.astype(np.uint8))
    # io.imsave(os.path.join(data_dir, 'match.png'), img_matches.astype(np.uint8))

    # for i in range(len(original_list)):
    #     b = original_list[i]
    #     if b in d[1]['train']:
    #         print(d[1]['train']['SMI_Benign/{}'.format(b)])
    #         d[1]['train']['SMI_Benign/{}'.format(b)] = 'SMI_Benign/{}'.format(replaced_list[i])
    #
    #     if b in d[1]['val']:
    #         print(d[1]['val']['SMI_Benign/{}'.format(b)])
    #         d[1]['val']['SMI_Benign/{}'.format(b)] = 'SMI_Benign/{}'.format(replaced_list[i])

