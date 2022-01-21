import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from collections import OrderedDict


def collect_metrics(dir, length=[0.2, 0.4, 0.8, 2]):
    res = OrderedDict({'acc': [], 'auc': [], 'f1-score': [], 'precision': [], 'recall': [], 'specificity': []})
    res_error = OrderedDict({'acc': [], 'auc': [], 'f1-score': [], 'precision': [], 'recall': [], 'specificity': []})

    for l in length:
        metrics_files = os.path.join(dir, 'seq_length_4_' + str(l), 'avg_metrics_best.json')
        with open(metrics_files, 'r') as f:
            metrics = json.load(f)

        res['acc'].append(metrics['accuracy'][0]*100)
        res_error['acc'].append(metrics['accuracy'][1]*100)

        res['auc'].append(metrics['auc'][0]*100)
        res_error['auc'].append(metrics['auc'][1]*100)

        res['f1-score'].append(metrics['f1-score'][0]*100)
        res_error['f1-score'].append(metrics['f1-score'][1]*100)

        res['precision'].append(metrics['precision'][0]*100)
        res_error['precision'].append(metrics['precision'][1]*100)

        res['recall'].append(metrics['recall'][0]*100)
        res_error['recall'].append(metrics['recall'][1]*100)

        res['specificity'].append(metrics['specificity'][0]*100)
        res_error['specificity'].append(metrics['specificity'][1]*100)

    return res, res_error


res_dir = '/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/cnn-diff-hc_MIC/with_rank_loss_weight'
res, res_error = collect_metrics(res_dir, length=[0.2, 0.4, 0.8, 2, 3])
# labels = ['seq_legth=2', 'seq_legth=3', 'seq_legth=4', 'seq_legth=5']

labels = ['weight=0.2', 'weight=0.4', 'weight=0.8', 'weight=2', 'without_ranking_loss']
acc = res['acc']
auc = res['auc']
f1_score = res['f1-score']
precision = res['precision']
recall = res['recall']
specificity = res['specificity']

acc_std = res_error['acc']
auc_std = res_error['auc']
f1_score_std = res_error['f1-score']
precision_std = res_error['precision']
recall_std = res_error['recall']
specificity_std = res_error['specificity']

x = np.arange(len(labels))  # the label locations
width = 0.50  # the width of the bars

# matplotlib.style.use('seaborn')
fig, ax = plt.subplots()
# rects1 = ax.bar(x-width/3, acc, width/3,  color='tan', edgecolor='black', label='accuracy', yerr=acc_std, capsize=2)
# rects2 = ax.bar(x, auc, width/3, color='teal', edgecolor='black', label='Auc', yerr=auc_std, capsize=2)
# rects3 = ax.bar(x + width/3, f1_score, width/3, color='salmon', edgecolor='black', label='f1-score', yerr=f1_score_std, capsize=2)
# ax.legend((rects1, rects2, rects3), ('accuracy', 'AUC', 'f1-score'), bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#        ncol=3, mode="expand", borderaxespad=0.)

# rects1 = ax.bar(x-width*2/5, acc, width/5,  color='tan', edgecolor='black', label='accuracy', yerr=acc_std, capsize=2)
# rects2 = ax.bar(x-width/5, auc, width/5, color='teal', edgecolor='black', label='Auc', yerr=auc_std, capsize=2)
# rects3 = ax.bar(x, f1_score, width/5, color='salmon', edgecolor='black', label='f1-score', yerr=f1_score_std, capsize=2)
# rects4 = ax.bar(x + width/5, precision, width/5, color='silver', edgecolor='black', label='precision', yerr=precision_std, capsize=2)
# rects5 = ax.bar(x + width*2/5, recall, width/5, color='plum', edgecolor='black', label='recall', yerr=recall_std, capsize=2)
# ax.legend((rects1, rects2, rects3, rects4, rects5), ('Accuracy', 'AUC', 'F1-score', 'Precision', 'Recall'), bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#        ncol=3, mode="expand", borderaxespad=0.)

rects1 = ax.bar(x-width*2/5, acc, width/5,  color='tan', edgecolor='black', label='accuracy', yerr=acc_std, capsize=2)
rects2 = ax.bar(x-width/5, auc, width/5, color='teal', edgecolor='black', label='Auc', yerr=auc_std, capsize=2)
rects3 = ax.bar(x, precision, width/5, color='salmon', edgecolor='black', label='precision', yerr=precision_std, capsize=2)
rects4 = ax.bar(x + width/5, recall, width/5, color='silver', edgecolor='black', label='recall', yerr=recall_std, capsize=2)
rects5 = ax.bar(x + width*2/5, specificity, width/5, color='plum', edgecolor='black', label='specificity', yerr=specificity_std, capsize=2)

ax.legend((rects1, rects2, rects3, rects4, rects5), ('Accuracy', 'AUC', 'Precision', 'Sensitivity', 'Specificity'), bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode="expand", borderaxespad=0., fontsize=14)

# ax.set_ylabel('evaluation metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.set_ylim([0, 100])
ax.tick_params(axis="y", labelsize=14)
fig.tight_layout()
plt.plot(res['auc'][:-1], ':')
# matplotlib.rc('font', size=10)
# plt.savefig(('/media/zyi/D8F29D09F29CED4E/Untitled Folder/cnn_pool_s_font.png'), dpi=600)
plt.show()
# # Add some text for labels, title and custom x-axis tick labels, etc.
#
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()



plt.plot([np.mean(wb), np.mean(wb), np.mean(wb), np.mean(wb)], 'g', linewidth=1.5)
# plt.fill_between(np.arange(4), [np.mean(wb) - np.std(wb), np.mean(wb)- np.std(wb), np.mean(wb)- np.std(wb), np.mean(wb)- np.std(wb)], [np.mean(wb) + np.std(wb), np.mean(wb)+ np.std(wb), np.mean(wb)+ np.std(wb), np.mean(wb)+ np.std(wb)], color='rebeccapurple', alpha=0.18)

plt.plot([np.mean(wbn), np.mean(wbn), np.mean(wbn), np.mean(wbn)], color='g', linestyle=':', linewidth=1.5)
# plt.fill_between(np.arange(4), [np.mean(wbn) - np.std(wbn), np.mean(wbn)- np.std(wbn), np.mean(wbn)- np.std(wbn), np.mean(wbn)- np.std(wbn)], [np.mean(wbn) + np.std(wbn), np.mean(wbn)+ np.std(wbn), np.mean(wbn)+ np.std(wbn), np.mean(wbn)+ np.std(wbn)], color='slateblue', alpha=0.18)

plt.plot([np.mean(gb), np.mean(gb), np.mean(gb), np.mean(gb)], 'coral', linewidth=1.5)
# plt.fill_between(np.arange(4), [np.mean(gb) - np.std(gb), np.mean(gb)- np.std(gb), np.mean(gb)- np.std(gb), np.mean(gb)- np.std(gb)], [np.mean(gb) + np.std(gb), np.mean(gb)+ np.std(gb), np.mean(gb)+ np.std(gb), np.mean(gb)+ np.std(gb)], color='salmon', alpha=0.18)

plt.plot([np.mean(gbn), np.mean(gbn), np.mean(gbn), np.mean(gbn)], color='coral', linestyle=':', linewidth=1.5)
# plt.fill_between(np.arange(4), [np.mean(gbn) - np.std(gbn), np.mean(gbn)- np.std(gbn), np.mean(gbn)- np.std(gbn), np.mean(gbn)- np.std(gbn)], [np.mean(gbn) + np.std(gbn), np.mean(gbn)+ np.std(gbn), np.mean(gbn)+ np.std(gbn), np.mean(gbn)+ np.std(gbn)], color='coral', alpha=0.18)

plt.legend(['weight_bce_fbn', 'weight_bce', 'general_bce_fbn', 'general_bce'])

plt.plot(wb, 'g', linewidth=1.5)
# plt.fill_between(np.arange(4), [np.mean(wb) - np.std(wb), np.mean(wb)- np.std(wb), np.mean(wb)- np.std(wb), np.mean(wb)- np.std(wb)], [np.mean(wb) + np.std(wb), np.mean(wb)+ np.std(wb), np.mean(wb)+ np.std(wb), np.mean(wb)+ np.std(wb)], color='rebeccapurple', alpha=0.18)

plt.plot(wbn, color='g', linestyle=':', linewidth=1.5)
# plt.fill_between(np.arange(4), [np.mean(wbn) - np.std(wbn), np.mean(wbn)- np.std(wbn), np.mean(wbn)- np.std(wbn), np.mean(wbn)- np.std(wbn)], [np.mean(wbn) + np.std(wbn), np.mean(wbn)+ np.std(wbn), np.mean(wbn)+ np.std(wbn), np.mean(wbn)+ np.std(wbn)], color='slateblue', alpha=0.18)

plt.plot(gb, 'coral', linewidth=1.5)
# plt.fill_between(np.arange(4), [np.mean(gb) - np.std(gb), np.mean(gb)- np.std(gb), np.mean(gb)- np.std(gb), np.mean(gb)- np.std(gb)], [np.mean(gb) + np.std(gb), np.mean(gb)+ np.std(gb), np.mean(gb)+ np.std(gb), np.mean(gb)+ np.std(gb)], color='salmon', alpha=0.18)

plt.plot(gbn, color='coral', linestyle=':', linewidth=1.5)
# plt.fill_between(np.arange(4), [np.mean(gbn) - np.std(gbn), np.mean(gbn)- np.std(gbn), np.mean(gbn)- np.std(gbn), np.mean(gbn)- np.std(gbn)], [np.mean(gbn) + np.std(gbn), np.mean(gbn)+ np.std(gbn), np.mean(gbn)+ np.std(gbn), np.mean(gbn)+ np.std(gbn)], color='coral', alpha=0.18)

plt.legend(['weight_bce_fbn: {:.2f}'.format(np.mean(wb)*100), 'weight_bce: {:.2f}'.format(np.mean(wbn)*100),
            'general_bce_fbn: {:.2f}'.format(np.mean(gb)*100), 'general_bce: {:.2f}'.format(np.mean(gbn)*100)])
plt.tight_layout()