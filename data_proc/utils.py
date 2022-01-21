import os
import cv2
import time
from glob import glob
import numpy as np
from PIL import Image
import skimage.io as io


def hair_removal(src_img):
    """
    :param src_img: uint8 array
    :return:
    """
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # for hair removal
    kernel = cv2.getStructuringElement(1, (17, 17))
    # Perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst_img = cv2.inpaint(src_img, thresh2, 1, cv2.INPAINT_TELEA)
    return dst_img


def skin_hair_removing(root_dir, result_dir):
    case_lists = glob(os.path.join(root_dir, 'SMI*', '*'))

    for case_dir in case_lists:
        image_list = glob(os.path.join(case_dir, '*MIC*'))

        start_time = time.time()
        new_case_dir = os.path.join(result_dir, case_dir.split('/')[-2], case_dir.split('/')[-1])
        if not os.path.exists(new_case_dir):
            os.makedirs(new_case_dir)

        for img_dir in image_list:
            img = io.imread(img_dir)
            img = hair_removal(img)

            io.imsave(os.path.join(new_case_dir, img_dir.split('/')[-1]), img)

        print('now processing ', case_dir, '\n', 'using time {:.3f}'.format(time.time() - start_time))


def resize(img):
    h, w, _ = img.shape
    if h>w:
        img = cv2.resize(img, (320, int(320*h/w)))
    else:
        img = cv2.resize(img, (int(320*w/h), 320))

    return img


if __name__ == '__main__':
    # root_dir = '/home/zyi/MedicalAI/new_serial_skin_data'
    # result_dir = '/home/zyi/MedicalAI/new_serial_skin_data/aa'
    # skin_hair_removing(root_dir, result_dir)

    root_dir = '/home/zyi/MedicalAI/HR_Serial_Skin_data'
    result_dir = '/home/zyi/MedicalAI/HR_Serial_Skin_data_320'
    cases = glob(os.path.join(root_dir, "*SMI*", '*'))
    for path in cases:
        img_list = glob(os.path.join(path, '*MIC*'))
        for i in img_list:
            img = io.imread(i)
            img = resize(img)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            save_path = i.replace('HR_Serial_Skin_data', 'HR_Serial_Skin_data_320')
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            io.imsave(save_path, img)