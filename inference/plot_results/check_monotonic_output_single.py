import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sn


exp_dir = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/CNN_features/single_img_cnn_ham'
length = [1, 2, 3, 4]
pos = []
neg = []
sn.set_style('darkgrid')

for length in [1, 2, 3, 4]:
    tmp_pos = []
    tmp_neg = []
    for fold in [0, 1, 2, 3, 4]:
        prediction = torch.load(os.path.join(exp_dir, 'seq_length_{}'.format(length),
                                             'fold_{}'.format(fold), 'test_res.pt'))

        pred = np.array(prediction['pred_score'])
        label = np.array(prediction['target'])

        tmp_pos.append(pred[label == 1].mean())
        tmp_neg.append(pred[label == 0].mean())

    pos.append(np.array(tmp_pos))
    neg.append(np.array(tmp_neg))

pos = np.array(pos).squeeze().transpose(1, 0)  # Nfold * Time
neg = np.array(neg).squeeze().transpose(1, 0)
print(pos.shape)
plt.figure(figsize=(4.8, 3.2))
plt.plot(np.arange(pos.shape[-1]), pos.mean(axis=0).squeeze(), color="#2E8B57", linewidth=1.8)
plt.plot(np.arange(neg.shape[-1]), neg.mean(axis=0).squeeze(), color="#DB7093", linewidth=1.8)
#
plt.fill_between(np.arange(pos.shape[-1]), pos.mean(axis=0) - pos.std(axis=0), pos.mean(axis=0) + pos.std(axis=0), color="#2E8B57", alpha=0.15)
plt.fill_between(np.arange(neg.shape[-1]), neg.mean(axis=0) - neg.std(axis=0), neg.mean(axis=0) + neg.std(axis=0), color="#DB7093", alpha=0.15)
plt.legend(['malignant', 'benign'], loc='lower left')
# plt.xlabel('time step', fontsize=12)
plt.ylabel('averaged prediction scores', fontsize=10)
# # plt.title('Predictions distribution of each time step', fontsize=12)
plt.xlim([-0.1, 3.2])
plt.ylim([0.2, 0.75])
plt.xticks([0, 1, 2, 3], ['Time 1', 'Time 2', 'Time 3', 'Time 4'], fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, 'pred_scores.png'), dpi=300)
plt.show()
