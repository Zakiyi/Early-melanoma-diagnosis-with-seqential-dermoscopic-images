import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import matplotlib
matplotlib.style.use('seaborn')
csv_file = pd.read_csv('/home/zyi/My_disk/train.csv')

patient_id = csv_file['patient_id'].array
patient_img = csv_file['image_name'].array
patient_label = csv_file['benign_malignant'].array
lesion_location = csv_file['anatom_site_general_challenge'].array
ids, count = np.unique(patient_id, return_counts=True)
# plt.plot(np.arange(len(ids)), count)
# plt.tight_layout()
#
#
# fig, ax = plt.subplots()
# plt.bar(np.arange(len(ids)), count)
# plt.xlabel('patients id')
# plt.ylabel('image nums')
# plt.title('ISIC2020 training data distribution')
#
# textstr = '\n'.join((
#     r'min image num: {}'.format(2),
#     r'max image num: {}'.format(115),
#     r'total patients: {}'.format(2056),
#     r'total image num: {}'.format(33126)))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.8, 0.95, textstr, transform=ax.transAxes, fontsize=10,
#         verticalalignment='top', horizontalalignment='left', bbox=props)
# plt.tight_layout()
#
# # plot with stacked benign malignant
# benign_count = []
# malignant_count = []
# for id in ids:
#     label = patient_label[patient_id==id]
#     benign_count.append(np.sum(label=='benign'))
#     malignant_count.append(np.sum(label=='malignant'))
#     assert benign_count[-1] + malignant_count[-1] == len(label)
#
# fig, ax = plt.subplots()
# plt.xlabel('patients id')
# plt.ylabel('image nums')
# plt.title('ISIC2020 training data distribution')
# plt.bar(np.arange(len(ids)), benign_count)
# plt.bar(np.arange(len(ids)), malignant_count)
#
# textstr = '\n'.join((
#     r'min image num: {}'.format(2),
#     r'max image num: {}'.format(115),
#     r'total patients: {}'.format(2056),
#     r'total image num: {}'.format(33126),
#     r'id_with malignant: {}'.format(428)))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.9, 0.95, textstr, transform=ax.transAxes, fontsize=10,
#         verticalalignment='top', horizontalalignment='left', bbox=props)
# plt.tight_layout()

# for melanoma
id_contain_malignant = np.unique(patient_id[patient_label == 'malignant'])
benign_count = []
malignant_count = []
for id in id_contain_malignant:
    label = patient_label[patient_id==id]
    benign_count.append(np.sum(label=='benign'))
    malignant_count.append(np.sum(label=='malignant'))
    if np.sum(label=='benign') == 0:
        print(id, 'benign is 0')
    if np.sum(label == 'malignant') == 0:
        print(id, 'malignant is 0')
    assert benign_count[-1] + malignant_count[-1] == len(label)

fig, ax = plt.subplots()
plt.xlabel('patients id')
plt.ylabel('image nums')
plt.title('ISIC2020 training data distribution of patient with melanoma')
plt.bar(np.arange(len(id_contain_malignant)), benign_count)
plt.bar(np.arange(len(id_contain_malignant)), malignant_count)

textstr = '\n'.join((
    r'min malignant image num: {}'.format(1),
    r'max malignant image num: {}'.format(8),
    r'total patient ids with melanoma: {}'.format(428),
    r'average melanoma num of each id: {}'.format(1.36),
    r'total image num: {}'.format(6927)))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.8, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left', bbox=props)
plt.tight_layout()
plt.show()
#
# get distribution of testing dataset
# csv_file = pd.read_csv('/home/zyi/My_disk/test.csv')
#
# patient_id = csv_file['patient_id'].array
# patient_img = csv_file['image_name'].array
# ids, count = np.unique(patient_id, return_counts=True)
# fig, ax = plt.subplots()
# plt.bar(np.arange(len(ids)), count)
# plt.xlabel('patients id')
# plt.ylabel('image nums')
# plt.title('ISIC2020 testing data distribution')
#
# textstr = '\n'.join((
#     r'min image num: {}'.format(3),
#     r'max image num: {}'.format(240),
#     r'total patients: {}'.format(690),
#     r'total image num: {}'.format(10982)))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.8, 0.95, textstr, transform=ax.transAxes, fontsize=10,
#         verticalalignment='top', horizontalalignment='left', bbox=props)
# plt.tight_layout()
#
#
#
# load some examples
# test_data_dir = '/home/zyi/My_disk/jpeg/train'
# id = id_contain_malignant[10]
# img_list = patient_img[patient_id == id]
# img_labels = patient_label[patient_id == id]
# lesion_loc = lesion_location[patient_id == id]
#
# assert len(img_list) == len(img_labels)
# id_dir = os.path.join('/home/zyi/My_disk/ISIC2020/malignant_ids', id)
# if not os.path.exists(id_dir):
#     os.makedirs(id_dir)
#
# for i in range(len(img_list)):
#     img_name = img_list[i]
#     try:
#         loc = lesion_loc[i].replace('/', '_')
#     except:
#         print(loc)
#     if img_labels[i] == 'malignant':
#         shutil.copy(os.path.join(test_data_dir, img_name + '.jpg'), os.path.join(id_dir, img_name + '_' + loc + '_malignant.jpg'))
#     elif img_labels[i] == 'benign':
#         shutil.copy(os.path.join(test_data_dir, img_name + '.jpg'), os.path.join(id_dir, img_name + '_' + loc + '_benign.jpg'))
#     else:
#         raise ValueError