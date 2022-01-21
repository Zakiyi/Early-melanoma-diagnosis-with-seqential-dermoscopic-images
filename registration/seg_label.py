import numpy as np
import json
import os
import skimage.io as io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def creat_img_mask(json_file, img_file):

    img = io.imread(img_file)
    with open(json_file, 'rb') as f:
        points = json.load(f)

    points = points['shapes'][0]['points']
    assert isinstance(points, list)

    # p1 = [points[0][0], points[1][1]]
    # p2 = [points[1][0], points[0][1]]
    # points.append(p2)
    # points.insert(1, p1)
    # convert list to tuple
    points = list(map(lambda x: tuple([x[0], x[1]]), points))  # [(x1, y1), (x2, y2), ....]

    mask = Image.new('L', (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    mask = np.array(mask)

    # mask = np.rot90(mask)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()
    masked_img = np.stack((mask, mask, mask), axis=-1) * img
    return mask, masked_img


if __name__ == '__main__':
    img_file = '/home/zyi/Desktop/examples/15600595/15600595_MIC_20120627.jpg'
    json_file = img_file.replace('jpg', 'json')

    mask, masked_img = creat_img_mask(json_file, img_file)
    io.imsave(img_file.replace('.jpg', '_seg.jpg'), masked_img)
    plt.imshow(masked_img)
    plt.show()