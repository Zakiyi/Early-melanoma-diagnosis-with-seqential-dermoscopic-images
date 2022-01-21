import os
import numpy as np
import shutil
from glob import glob
import pickle as pkl


with open('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/data_setting/data_split.pkl', 'rb') as f:
    split = pkl.load(f)


train = split[1]['train']
val = split[1]['val']

data_root = '/home/zyi/MedicalAI/HR_Diff_data/Aligned'
# for case in train:
#     img_list = glob(os.path.join(data_root, case, '*.png'))
#     for img in img_list:
#         if 'Benign' in img:
#             dst_path = '/home/zyi/MedicalAI/HR_Diff_data/diff_dataset/train/benign/' + case.split('/')[-1] + '_{}'.format(os.path.basename(img))
#         else:
#             dst_path = '/home/zyi/MedicalAI/HR_Diff_data/diff_dataset/train/malignant/' + case.split('/')[
#                 -1] + '_{}'.format(os.path.basename(img))
#         shutil.copy(img, dst_path)

# for case in val:
#     img_list = glob(os.path.join(data_root, case, '*.png'))
#     for img in img_list:
#         if 'Benign' in img:
#             dst_path = '/home/zyi/MedicalAI/HR_Diff_data/diff_dataset/val/benign/' + case.split('/')[-1] + '_{}'.format(os.path.basename(img))
#         else:
#             dst_path = '/home/zyi/MedicalAI/HR_Diff_data/diff_dataset/val/malignant/' + case.split('/')[-1] + '_{}'.format(os.path.basename(img))
#         shutil.copy(img, dst_path)

aaa = '/home/zyi/MedicalAI/HR_Diff_data/diff_dataset/train'
img_list = glob(os.path.join(aaa, '*', "*"))

