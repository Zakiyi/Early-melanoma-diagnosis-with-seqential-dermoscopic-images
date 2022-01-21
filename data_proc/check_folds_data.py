import pickle as pkl
import os
import shutil
from glob import glob

data_root = '/home/zyi/MedicalAI/Serial_Skin_data'
data_split_file = '/run_exp/data_split.pkl'
dest_file_dir = '/home/zyi/Desktop/data_folds'

with open(data_split_file, 'rb') as f:
    data_split = pkl.load(f)


for fold in range(5):
    train = data_split[fold]['train']
    val = data_split[fold]['val']

    train_dir = os.path.join(dest_file_dir, 'fold_' + str(fold), 'train')
    val_dir = os.path.join(dest_file_dir, 'fold_' + str(fold), 'val')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    train_img_length = []
    for t in train:
        img_list = glob(os.path.join(data_root, t, '*MIC*'))
        img_dir = os.path.join(train_dir, t.split('/')[-1])
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        list(map(lambda x: shutil.copy(x, img_dir), img_list))
        train_img_length.append(len(img_list))

    # print(fold, 'train length:\n', train_img_length)
    val_img_length = []
    for v in val:
        img_list = glob(os.path.join(data_root, v, '*MIC*'))
        img_dir = os.path.join(val_dir, v.split('/')[-1])

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        list(map(lambda x: shutil.copy(x, img_dir), img_list))
        val_img_length.append(len(img_list))

    # print(fold, 'val length:\n', val_img_length)