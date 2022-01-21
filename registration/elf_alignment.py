import torch
import cv2
import numpy as np
import skimage.io as io
import skimage.transform as tf
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn.functional as fun
from  registration.helper.utils import postproc, SuperPoint_interpolate
from models.model_uitls import load_pretrained_resnet
import colorcorrect.algorithm as cca
from skimage.measure import ransac
from skimage.transform import EuclideanTransform, warp


def feature_detect(img, model):
    # img = cca.grey_world(img)
    input = tf.resize(img, (320, 320))
    # mean = input.mean(axis=1).mean(axis=1)

    mean = np.mean(input, axis=(0, 1, 2))
    size = np.prod(np.array(input.shape))
    stddev = np.maximum(np.std(input, axis=(0, 1, 2)), 1.0 / np.sqrt(size * size))
    input = (input - mean) / stddev
    # print(mean)
    # print(stddev)

    input = input.transpose(2, 0, 1)
    input = torch.from_numpy(input).float()
    # input = F.normalize(input, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    input = input.unsqueeze(dim=0)
    input.requires_grad = True

    final, features = model(input)
    # final.backward()
    f = features[1]
    f.backward(f)

    img_grad = input.grad.squeeze().numpy().transpose(1, 2, 0).astype(np.float)
    ptf, pts, vis_map = postproc(img_grad)
    vis_map = vis_map[:, :, ::-1]
    plt.imshow(vis_map)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=1)
    plt.figure()
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float) + 0.8*vis_map.astype(np.float)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=1)

    kp = []
    for p in pts.T:
        k = cv2.KeyPoint(x=p[0], y=p[1], _size=4, _angle=0, _response=0, _octave=0, _class_id=0)
        kp.append(k)
    #
    # img = cv2.resize(img, (320, 320))
    # img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kp_on_img = np.tile(np.expand_dims(img0, 2), (1, 1, 3))
    # for i, k in enumerate(kp):
    #     pt = (int(round(k.pt[0])), int(round(k.pt[1])))
    #     cv2.circle(kp_on_img, pt, 4, (0, 255, 0), -1, lineType=16)

    # plt.imshow(kp_on_img)
    # plt.show()

    # using KAZE detector
    # detector = cv2.AKAZE_create(descriptor_size=0, threshold=0.0008, nOctaves=4)
    #
    # # find the keypoints and compute the descriptors with ORB
    # kp1, des1 = detector.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    #
    # kp = [[x.pt[0], x.pt[1], 0] for x in kp1]
    # pts = np.array(kp)
    # pts = pts.reshape(3, -1)

    des_coarse = features[2].squeeze().detach().numpy().transpose(1, 2, 0)  # H*W*C
    des = SuperPoint_interpolate(pts, des_coarse, 320, 320)

    return kp, des


def feature_matching(img1, kp1, des1, img2, kp2, des2):
    img1 = cv2.resize(img1, (320, 320))
    img2 = cv2.resize(img2, (320, 320))

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    good = []
    matches = matcher.knnMatch(des1, des2, k=2)
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.96 * n.distance:
            good.append(m)

    match_des_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(match_des_img)
    plt.tight_layout()
    plt.axis('off')
    plt.show(block=0)
    kp1 = [[x.pt[0], x.pt[1]] for x in kp1]
    kp2 = [[x.pt[0], x.pt[1]] for x in kp2]

    matches = np.array([[m.queryIdx, m.trainIdx] for m in good])
    src_points = np.array([kp1[i] for i in matches[:, 0]])  # N * 2
    des_points = np.array([kp2[i] for i in matches[:, 1]])  # N *2

    model, inliers = ransac((src_points, des_points), EuclideanTransform,
                            min_samples=2, residual_threshold=2, max_trials=5000)

    return model


if __name__ == '__main__':

    model = load_pretrained_resnet('resnet34')
    img1 = io.imread('/home/zyi/Desktop/examples/14450763/14450763_MIC_20081014.jpg')
    img2 = io.imread('/home/zyi/Desktop/examples/14450763/14450763_MIC_20100301.jpg')
    kp1, des1 = feature_detect(img1, model)
    kp2, des2 = feature_detect(img2, model)
    trans_model = feature_matching(img1, kp1, des1, img2, kp2, des2)

    img1 = cv2.resize(img1, (320, 320))
    img2 = cv2.resize(img2, (320, 320))

    img_warp = warp(img1, trans_model.inverse, output_shape=img1.shape, mode='reflect')
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(img1)
    axes[0].set_axis_off()
    axes[1].imshow(img_warp)
    axes[1].set_axis_off()
    axes[2].imshow(img2)
    axes[2].set_axis_off()
    plt.tight_layout()
    plt.show()


