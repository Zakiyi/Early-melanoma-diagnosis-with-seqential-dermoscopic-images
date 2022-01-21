from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import transform as tf
from skimage.measure import ransac
from skimage.transform import EuclideanTransform, EssentialMatrixTransform
import skimage.io as io
import cv2
import math
from skimage import data
import numpy as np
from skimage.restoration import denoise_bilateral
"""
reference: scikit-image Fundamental matrix estimation
https://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html#sphx-glr-auto-examples-transform-plot-fundamental-matrix-py
"""


def img_alignment(img1, img2, transformation=tf.EuclideanTransform, show_res=0):
    # img1 = cv2.bilateralFilter(img1, 10, 25, 20)
    # img2 = cv2.bilateralFilter(img2, 10, 25, 20)
    # tform = EuclideanTransform(rotation=0.2, translation=(20, -10))
    # img2 = tf.warp(img1, tform.inverse, output_shape=img1.shape)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    des_extractor = ORB(n_keypoints=100, fast_n=11, fast_threshold=0.08, harris_k=0.12)
    des_extractor.detect_and_extract(rgb2gray(img1))
    kp1 = des_extractor.keypoints  # N*2 array
    des1 = des_extractor.descriptors  # N*256 array

    des_extractor = ORB(n_keypoints=100, fast_n=11, fast_threshold=0.08, harris_k=0.12)
    des_extractor.detect_and_extract(rgb2gray(img2))
    kp2 = des_extractor.keypoints
    des2 = des_extractor.descriptors

    matches = match_descriptors(des1, des2, cross_check=True)  # N*2 array
    # kp1_list = list(map(lambda x: cv2.KeyPoint(x[1], x[0], 0), [list(kp1)[i] for i in matches[:, 0]]))
    # img1 = cv2.drawKeypoints(img1, kp1_list, None, color=(0, 255, 0), flags=0)
    # kp2_list = list(map(lambda x: cv2.KeyPoint(x[1], x[0], 0), list(kp2)))
    # img2 = cv2.drawKeypoints(img2, kp2_list, None, color=(0, 255, 0), flags=0)
    #
    fig, axes = plt.subplots(1, 1)
    plot_matches(axes, img1, img2, kp1, kp2, matches, keypoints_color='b', only_matches=True)
    plt.axis('off')
    plt.title("Original Image vs. Transformed Image")
    plt.show(block=0)

    src = kp1[matches[:, 0]]
    des = kp2[matches[:, 1]]
    print(src.shape)
    model, inliers = ransac((src[:, ::-1], des[:, ::-1]),
                            transformation, min_samples=2,
                            residual_threshold=2, max_trials=5000)
    return model
# print('matrix: ', model.params)
# print('rotation: ', model.rotation)
# print('translation: ', model.translation)


if __name__ == '__main__':

    img1 = io.imread('/home/zyi/Desktop/examples/15600595/15600595_MIC_20120326.jpg')
    img2 = io.imread('/home/zyi/Desktop/examples/15600595/15600595_MIC_20120627.jpg')
    img1 = cv2.resize(img1, (400, 320))
    model = img_alignment(img1, img2)

    img1 = io.imread('/home/zyi/Desktop/examples/15600595/15600595_MIC_20120326.jpg')
    img2 = io.imread('/home/zyi/Desktop/examples/15600595/15600595_MIC_20120627.jpg')
    img_warp = tf.warp(img1, model.inverse, output_shape=img1.shape, mode='reflect')
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(img1)
    axes[1].imshow(img_warp)
    axes[2].imshow(img2)
    plt.show()
