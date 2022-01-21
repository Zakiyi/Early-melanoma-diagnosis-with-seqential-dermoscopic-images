import os
import torch
import numpy as np
import seaborn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from _collections import OrderedDict
from sklearn.metrics import roc_curve, confusion_matrix


def get_reviewer_prediction(csv, R='R1'):
    res = OrderedDict({'scores': [], 'targets': []})
    diagnose_res = csv['diagnose_result']
    return res


# csv = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
#                   'human_results/review_temp1.csv')

# R1 = get_reviewer_prediction(csv, R='R1')
# R2 = get_reviewer_prediction(csv, R='R2')
# R3 = get_reviewer_prediction(csv, R='R3')
# R4 = get_reviewer_prediction(csv, R='R4')
# R5 = get_reviewer_prediction(csv, R='R5')
# R6 = get_reviewer_prediction(csv, R='R6')
# R7 = get_reviewer_prediction(csv, R='R7')
# R8 = get_reviewer_prediction(csv, R='R8')
# R9 = get_reviewer_prediction(csv, R='R9')
# R10 = get_reviewer_prediction(csv, R='R10')
# R11 = get_reviewer_prediction(csv, R='R11')
# R12 = get_reviewer_prediction(csv, R='R12')

R1 = [0.9340659341, 0.2087912088]
R2 = [0.6593406593, 0.5824175824]
R3 = [0.6703296703, 0.5384615385]
R4 = [0.6043956044, 0.6593406593]
R5 = [0.7692307692, 0.3076923077]
R6 = [0.4945054945, 0.7582417582]
R7 = [0.8021978022, 0.2527472527]
R8 = [0.8791208791, 0.2857142857]
R9 = [0.8131868132, 0.4065934066]
R10 = [0.7252747253, 0.5384615385]
R11 = [0.7142857143, 0.5384615385]
R12 = [0.8131868132, 0.2417582418]


def plot_roc_curve(pred_score, target, pred_label):
    matplotlib.style.use('seaborn')
    fpr, tpr, thresholds = roc_curve(target, pred_score)

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 + fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    roc_t_threshold = list(roc_t['threshold'])[0]
    pred_label = list(map(lambda x: 1 if x > roc_t_threshold else 0, pred_score))

    accuracy = metrics.accuracy_score(target, pred_label)
    tn, fp, fn, tp = confusion_matrix(target, pred_label).ravel()
    specificity = tn / (tn + fp)
    recall = metrics.recall_score(target, pred_label)
    auc = metrics.roc_auc_score(target, pred_score)
    print('auc ', auc)
    print('acc ', accuracy)
    print('spec ', specificity)
    print('sens ', recall)
    return fpr, tpr, recall, specificity


root = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/CST features/SCA/seq_length_4'

T1_res = OrderedDict({'scores': [], 'targets': [], 'pred_labels': []})
T2_res = OrderedDict({'scores': [], 'targets': [], 'pred_labels': []})
T3_res = OrderedDict({'scores': [], 'targets': [], 'pred_labels': []})
T4_res = OrderedDict({'scores': [], 'targets': [], 'pred_labels': []})

# load prediction score from all folds
for fold in [0, 1, 2, 3, 4]:
    eval_res = torch.load(os.path.join(root, 'fold_{}/exit_scores_eval.pt'.format(fold)))
    T1_res['scores'].extend(eval_res['time_3']['pred_scores'])
    T1_res['targets'].extend(eval_res['time_3']['targets'])
    T1_res['pred_labels'].extend(eval_res['time_3']['pred_labels'])

fpr, tpr, recall, specificity = plot_roc_curve(T1_res['scores'], T1_res['targets'],  T1_res['pred_labels'])

f, ax = plt.subplots(figsize=(6, 4))
plt.plot(fpr, tpr)
our_model = plt.scatter(1-specificity, recall, marker='d', color='k', label='our_model')
R1_ = plt.scatter(1-R1[1], R1[0], marker='o', label='R1')
R2_ = plt.scatter(1-R2[1], R2[0], marker='o', label='R2')
R3_ = plt.scatter(1-R3[1], R3[0], marker='o', label='R3')
R4_ = plt.scatter(1-R4[1], R4[0], marker='o', label='R4')
R5_ = plt.scatter(1-R5[1], R5[0], marker='o', label='R5')
R6_ = plt.scatter(1-R6[1], R6[0], marker='o', label='R6')
R7_ = plt.scatter(1-R7[1], R7[0], marker='o', label='R7')
R8_ = plt.scatter(1-R8[1], R8[0], marker='o', label='R8')
R9_ = plt.scatter(1-R9[1], R9[0], marker='o', label='R9')
R10_ = plt.scatter(1-R10[1], R10[0], marker='o', label='R10')
R11_ = plt.scatter(1-R11[1], R11[0], marker='o', label='R11')
R12_ = plt.scatter(1-R12[1], R12[0], marker='o', label='R12')

plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
# handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=[R1_, R2_, R3_, R4_, R5_, R6_, R7_, R8_, R9_, R10_, R11_, R12_,our_model])
plt.tight_layout()
# for fold in [0, 1, 2, 3, 4]:
#     eval_res = torch.load(os.path.join(root, 'fold_{}/exit_scores_eval.pt'.format(fold)))
#     T2_res['scores'].extend(eval_res['time_1']['pred_scores'])
#     T2_res['targets'].extend(eval_res['time_1']['targets'])
# fpr, tpr, recall, specificity = plot_roc_curve(T2_res['scores'], T2_res['targets'])
# plt.plot(fpr, tpr)
#
# for fold in [0, 1, 2, 3, 4]:
#     eval_res = torch.load(os.path.join(root, 'fold_{}/exit_scores_eval.pt'.format(fold)))
#     T3_res['scores'].extend(eval_res['time_2']['pred_scores'])
#     T3_res['targets'].extend(eval_res['time_2']['targets'])
# fpr, tpr, recall, specificity = plot_roc_curve(T3_res['scores'], T3_res['targets'])
# plt.plot(fpr, tpr)
#
# for fold in [0, 1, 2, 3, 4]:
#     eval_res = torch.load(os.path.join(root, 'fold_{}/exit_scores_eval.pt'.format(fold)))
#     T4_res['scores'].extend(eval_res['time_3']['pred_scores'])
#     T4_res['targets'].extend(eval_res['time_3']['targets'])
# fpr, tpr, recall, specificity = plot_roc_curve(T4_res['scores'], T4_res['targets'])
# plt.plot(fpr, tpr)
plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/time0.png', dpi=300)
plt.show()

