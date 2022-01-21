import os
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


matplotlib.style.use('seaborn')
csv = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/diffcult0.csv')
csv = csv.sort_values(by='dft', ascending=True)

csv_data = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/diffcult0.csv')
d = csv_data.loc[:, '1R':'AI']
d = d.iloc[:, ::-1]

csv_label = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/diffcult_label.csv')
l = csv_label.loc[:, '1R':'AI']
l = l.iloc[:, ::-1]
f, ax = plt.subplots(figsize=(24, 6))
sb.heatmap(d.transpose(), annot=False, fmt='', linewidths=0.1, cbar=True,
           cbar_kws={'orientation': 'horizontal', 'fraction': 0.05, 'pad': 0.08},
           cmap=['#eb8f8f', '#ed6663',  '#bbd196', '#91d18b']) #'#2FC4B2''#91D18B' ])#'#28DF99'])

cbar = ax.collections[0].colorbar
cbar.set_ticks([1.4, 2.1, 2.9, 3.6])
cbar.set_ticklabels(['FN', 'FP', 'TN', 'TP'])
#plt.text(68, 15, 'TP', zorder=1)

#plt.text(68, 12, "spam", size=5,
#         ha="right", va="top",
#         bbox=dict(boxstyle="square",
#                   ec=(1., 0.5, 0.5),
#                   fc=(1., 0.8, 0.8),
#                   )
#         )
plt.tight_layout()
plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/diffcult0.png', dpi=1200)
plt.show()