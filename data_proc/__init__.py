import torch


def padding_diff_img_seq(padding_index, images, diff_images):
    """

    :param padding_index:
    :param images: T*C*H*W
    :param diff_images: T-1*C*H*W
    :return:
    """
    diff_img_list = []
    for i in range(len(padding_index) - 1):
        if padding_index[:-1][i] != 0:
            diff_img_list.append(diff_images[i, ...])

    if len(diff_img_list) != len(padding_index) - 1:
        if len(diff_img_list) == 0:
            diff_images = images[1:, ...]
        else:
            for j in range(len(padding_index) - 1 - len(diff_img_list)):
                # diff_img_list.insert(0, diff_img_list[0])
                diff_img_list.insert(0, images[j + 1, ...])
            diff_images = torch.stack(diff_img_list, dim=0)

    return diff_images


def padding_img_seq(padding_index, img_sequence):
    """
    :param padding_index: [0, 0, 1, 0...] index of image if padded image or not
    :param img_sequence: {'images': T*C*H*W, 'diff_images': T-1*C*H*W}
    :return:
    """
    assert len(img_sequence['images'].shape) == 4 or len(img_sequence['images'].shape) == 5

    if len(img_sequence['images'].shape) == 4:
        img_sequence['diff_images'] = padding_diff_img_seq(padding_index, img_sequence['images'],
                                                           img_sequence['diff_images'])

    else:
        for i in range(img_sequence['images'].shape[0]):
            img_sequence['diff_images'][i, ...] = padding_diff_img_seq(padding_index, img_sequence['images'][i, ...],
                                                                       img_sequence['diff_images'][i, ...])
    return img_sequence
