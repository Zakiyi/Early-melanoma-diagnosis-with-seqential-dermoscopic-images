import matplotlib.pyplot as plt

'''
--------------------------------------
------------- Dice plot --------------
--------------------------------------
'''
kidney = [[96.58, 97.40, 97.21], [97.15, 97.45, 97.27]]
tumor = [[80.93, 85.31, 82.66], [83.97, 82.67, 83.02]]
composite = [[88.75, 91.35, 89.93], [90.56, 90.06, 90.15]]

fig, axes = plt.subplots(1, 3)

m = axes[0].boxplot(kidney,  patch_artist=True)
axes[0].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[0].set_title('kidney dice')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

m=axes[1].boxplot(tumor,  patch_artist=True)
axes[1].set_yticklabels([81.0, 81.0, 82.0, 83.0, 84.0, 85.0])
axes[1].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[1].set_title('tumor dice')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

m=axes[2].boxplot(composite,  patch_artist=True)
axes[2].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[2].set_title('composite dice')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

figure = plt.gcf()
figure.subplots_adjust(left=0.06)
figure.subplots_adjust(right=0.97)
figure.subplots_adjust(bottom=0.07)
figure.subplots_adjust(top=0.92)
figure.set_figheight(4)
figure.set_figwidth(8)
plt.show(block=0)
plt.savefig('/home/zyi/ISBI2020/box_dice.svg', format='svg', dpi=1200)

'''
--------------------------------------
------------- Precision plot --------------
--------------------------------------
'''

kidney = [[96.27, 97.46, 96.97], [97.19, 97.51, 97.10]]
tumor = [[87.78, 87.27, 87.43], [91.48, 87.59, 88.85]]
composite = [[92.02, 92.36, 92.20], [94.34, 92.55, 92.98]]

fig, axes = plt.subplots(1, 3)

m = axes[0].boxplot(kidney,  patch_artist=True)
axes[0].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[0].set_title('kidney precision')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

m=axes[1].boxplot(tumor,  patch_artist=True)
axes[1].set_yticklabels([88.0, 88.0, 89.0, 90.0, 91.0])
axes[1].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[1].set_title('tumor precision')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

m=axes[2].boxplot(composite,  patch_artist=True)
axes[2].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[2].set_title('composite precision')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

figure = plt.gcf()
figure.subplots_adjust(left=0.06)
figure.subplots_adjust(right=0.97)
figure.subplots_adjust(bottom=0.07)
figure.subplots_adjust(top=0.92)
figure.set_figheight(4)
figure.set_figwidth(8)
plt.show(block=0)
plt.savefig('/home/zyi/ISBI2020/box_precision.svg', format='svg', dpi=1200)
'''
--------------------------------------
------------- recall plot ------------
--------------------------------------
'''

kidney = [[97.08, 97.49, 97.50], [97.13, 97.41, 97.46]]
tumor = [[81.64, 86.31, 82.55], [82.09, 82.52, 82.03]]
composite = [[89.36, 91.90, 90.02], [89.61, 89.97, 89.75]]

fig, axes = plt.subplots(1, 3)

m = axes[0].boxplot(kidney,  patch_artist=True)
axes[0].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[0].set_title('kidney recall')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

m=axes[1].boxplot(tumor,  patch_artist=True)
axes[1].set_yticklabels([82.0, 82.0, 83.0, 84.0, 85.0, 86.0])
axes[1].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[1].set_title('tumor recall')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

m=axes[2].boxplot(composite,  patch_artist=True)
axes[2].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[2].set_title('composite recall')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

figure = plt.gcf()
figure.subplots_adjust(left=0.06)
figure.subplots_adjust(right=0.97)
figure.subplots_adjust(bottom=0.07)
figure.subplots_adjust(top=0.92)
figure.set_figheight(4)
figure.set_figwidth(8)
plt.show(block=0)
plt.savefig('/home/zyi/ISBI2020/box_recall.svg', format='svg', dpi=1200)

'''
--------------------------------------
------------- jaccard plot ------------
--------------------------------------
'''

kidney = [[93.54, 95.07, 94.60], [94.46, 95.05, 94.69]]
tumor = [[72.10, 76.23, 73.47], [75.45, 73.34, 73.98]]
composite = [[82.82, 85.65, 84.03], [84.96, 84.20, 84.33]]

fig, axes = plt.subplots(1, 3)

m = axes[0].boxplot(kidney,  patch_artist=True)
axes[0].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[0].set_title('kidney jaccard')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

m=axes[1].boxplot(tumor,  patch_artist=True)
axes[1].set_yticklabels([72.0, 72.0, 73.0, 74.0, 75.0, 76.0])
axes[1].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[1].set_title('tumor jaccard')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

m=axes[2].boxplot(composite,  patch_artist=True)
axes[2].set_xticklabels(['Basic UNet', 'SCFI UNet'])
axes[2].set_title('composite jaccard')
colors = ['lightblue', 'tan']
for patch, color in zip(m['boxes'], colors):
    patch.set_facecolor(color)

figure = plt.gcf()
figure.subplots_adjust(left=0.06)
figure.subplots_adjust(right=0.97)
figure.subplots_adjust(bottom=0.07)
figure.subplots_adjust(top=0.92)
figure.set_figheight(4)
figure.set_figwidth(8)
plt.show(block=0)
plt.savefig('/home/zyi/ISBI2020/box_jaccard.svg', format='svg', dpi=1200)