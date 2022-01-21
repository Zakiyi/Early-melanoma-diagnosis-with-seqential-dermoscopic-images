import numpy as np
import cv2
import os
import time
from glob import glob
from matplotlib import pyplot as plt
from PIL import Image
import skimage.io as io
from skimage.measure import ransac
from skimage.transform import EuclideanTransform, warp
from skimage.feature import match_descriptors
from colorcorrect.util import to_pil
import colorcorrect.algorithm as cca


class Akaze_alignmenter():
    def __init__(self, data_folder, save_dir, size):
        """
        :param data_folder: folder of input data
        :param save_dir: folder for saving output data
        """
        self.size = size
        self.out_dir = save_dir
        self.data_root = data_folder
        self.data_list = glob(os.path.join(data_folder, '*SMI*', '*'))
        print('Total {} data found!!!'.format(len(self.data_list)))

    def convert_points(self, kp1, kp2, matches):
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

    def hair_removal(self, src):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # for hair removal
        kernel = cv2.getStructuringElement(1, (17, 17))
        # Perform the blackHat filtering on the grayscale image to find the
        # hair countours
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
        return dst

    def alignment_fun(self, moving_img, ref_img, threshold=0.0001, plot_matches=False):
        """
        :param img1: RGB image in numpy array (reference image or destination image)
        :param img2:                          (moving image or source image)
        :param plot_only_inliers:
        :return:
        """
        print('threshold ', threshold)
        if moving_img.shape != ref_img.shape:
            print('image pair have different shape!!!', moving_img.shape, ref_img.shape)
            moving_img = cv2.resize(moving_img, (ref_img.shape[1], ref_img.shape[0]))

        # Initiate KAZE detector
        detector = cv2.AKAZE_create(descriptor_size=0, threshold=threshold, nOctaves=4)  # 0.0008

        # find the keypoints and compute the descriptors
        kp1, des1 = detector.detectAndCompute(cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = detector.detectAndCompute(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), None)

        # match the keypoint according to its features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) <= 2:
            raise IndexError

        moving_points, ref_points = self.convert_points(kp1, kp2, matches)

        # estimate transformation parameters
        model, inliers = ransac((moving_points, ref_points), EuclideanTransform,
                                min_samples=2, residual_threshold=2, max_trials=5000)

        if len(inliers) <= 2:
            raise IndexError

        warpped_img = warp(moving_img, model.inverse, output_shape=moving_img.shape, mode='reflect', order=1)

        if plot_matches:
            m = []
            for k in range(len(inliers)):
                if inliers[k]:
                    m.append(matches[k])

            img_matches = cv2.drawMatches(moving_img, kp1, ref_img, kp2, m, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # else:
        #     img_matches = cv2.drawMatches(moving_img, kp1, ref_img, kp2, matches, None,
        #                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.figure()
            plt.imshow(img_matches)
            plt.axis('off')
            plt.show(block=0)
            plt.tight_layout()

            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(moving_img)
            axes[1].imshow(warpped_img)
            axes[2].imshow(ref_img)

            plt.show(block=1)

        return warpped_img

    def resize_img(self, img, ):
        h, w = img.shape[:2]

        if h < w:
            img = cv2.resize(img, (int(self.size*w/h), self.size))
        else:
            img = cv2.resize(img, (self.size, int(self.size*h/w)))

        # if hair_rm:
        img = self.hair_removal(np.array(img))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img

    def img_seq_align(self, img_list):
        if len(img_list) > 1:
            ref_img = self.resize_img(io.imread(img_list[0]))
            img_out_dir = os.path.join(self.out_dir, *img_list[0].split('/')[-3:])

            if not os.path.exists(os.path.dirname(img_out_dir)):
                os.makedirs(os.path.dirname(img_out_dir))

            io.imsave(img_out_dir, ref_img)
            for i in range(1, len(img_list)):
                mov_img = self.resize_img(io.imread(img_list[i]))
                # mov_img = mov_img[:, :-26, :]
                # # mov_img = self.resize_img(mov_img)
                # print(mov_img.shape)
                # try:
                #     warpped_img = self.alignment_fun(mov_img, ref_img, threshold=0.0008)
                # except IndexError:
                #     warpped_img = self.alignment_fun(mov_img, ref_img, threshold=0.00015)
                #
                # if np.max(warpped_img) == 0:
                warpped_img = self.alignment_fun(mov_img, ref_img, threshold=0.00006)
                ref_img = cv2.normalize(warpped_img, None, 0, 255, cv2.NORM_MINMAX)
                ref_img = ref_img.astype(np.uint8)
                # ref_img = ref_img[:, :-30, :]
                ref_img = cv2.resize(ref_img, (400, 320))
                img_out_dir = os.path.join(self.out_dir, *img_list[i].split('/')[-3:])
                io.imsave(img_out_dir, ref_img)
        else:
            ref_img = self.resize_img(io.imread(img_list[0]))
            img_out_dir = os.path.join(self.out_dir, *img_list[0].split('/')[-3:])
            if not os.path.exists(os.path.dirname(img_out_dir)):
                os.makedirs(os.path.dirname(img_out_dir))

            io.imsave(img_out_dir, ref_img)

    def run(self):
        start_time = time.time()
        for i in range(len(self.data_list)):
            img_list = glob(os.path.join(self.data_list[i], '*MIC*'))
            img_list = sorted(img_list, key=lambda x: x.split('_')[-1].split('.')[0])
            assert len(img_list) > 0

            self.img_seq_align(img_list)
            print('{}-th sample alignment achieved, using time {:.2f}s !!!'.format(i, time.time()-start_time))


if __name__ == '__main__':

    input_folder = '/home/zyi/MedicalAI/Original skin data/replaced_benign/replaced'
    output_folder = '/home/zyi/MedicalAI/Original skin data/replaced_benign/replaced/ssss'

    alignmenter = Akaze_alignmenter(input_folder, output_folder, size=320)
    alignmenter.run()