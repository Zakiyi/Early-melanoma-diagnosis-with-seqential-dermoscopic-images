import os
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('seaborn')
training_result_dirs = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/TSA/sep_train_batchmean/tkd_dst_klv_avg/round_0'
seq_length = [4]
plt.figure(figsize=(6, 3.4))
dst = ['bce_base', 'dst_bce_0.1_0.9', 'dst_bce_0.2_0.8', 'dst_bce_0.3_0.7', 'dst_bce_0.4_0.6', 'dst_bce_0.5_0.5']
# for dir in glob(os.path.join(training_result_dirs, 'd*')):
for dir in dst:
    dir = os.path.join(training_result_dirs, dir)
    # legends.append(dir.split('/')[-1])+

    for length in seq_length:
        res_dir = os.path.join(dir, 'seq_length_{}'.format(length), 'time_metrics.json')
        print(res_dir)
        with open(res_dir, 'rb') as f:
            m = json.load(f)
        res = np.array([m['time_0']['auc'][0], m['time_1']['auc'][0], m['time_2']['auc'][0], m['time_3']['auc'][0]])
        plt.plot(res, linewidth=1.8)

legends = [r'$\alpha=0.$', r'$\alpha=0.1$', r'$\alpha=0.2$', r'$\alpha=0.3$', r'$\alpha=0.4$', r'$\alpha=0.5$']
plt.legend(legends)
plt.ylim([67.5, 73])
plt.ylabel('AUC')
plt.xticks([0, 1, 2, 3], ['Time 1', 'Time 2', 'Time 3', 'Time 4'])
# plt.savefig(os.path.join(training_result_dirs, 'res4.png'))
plt.tight_layout()
plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/ablation_results/auc_dst.png', dpi=300)
plt.show()
