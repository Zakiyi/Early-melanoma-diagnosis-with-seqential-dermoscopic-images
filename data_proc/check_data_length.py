import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt

data_root = '/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned'
case_list = glob(os.path.join(data_root, 'SMI*', '*'))

img_lengths = []
for path in case_list:
    img_list = glob(os.path.join(path, '*MIC*'))
    img_num = len(img_list)
    img_lengths.append(img_num)

img_lengths = np.array(img_lengths)
a, b = np.unique(img_lengths, return_counts=True)
fig, ax = plt.subplots()
rects = ax.bar(a, b)


"""Attach a text label above each bar in *rects*, displaying its height."""
for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
plt.show()