import os
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt


sns.set(style="whitegrid")
## template
# image_idx = np.arange(1, 91)
# evaluator = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12']
# image_num = np.tile([9, 6, 5, 8, 4], 18)
#
# images_idx = np.repeat(image_idx, len(evaluator))
# images_num = np.repeat(image_num, len(evaluator))
# evaluator = np.tile(evaluator, len(image_idx))
#
# model_diagnose = np.tile([7, 5, 4, 6, 3], 18)
# model_diagnose = np.repeat(model_diagnose, 10)
#
# diagnose_date = np.concatenate([np.random.randint(1, 10, 10), np.random.randint(1, 7, 10),
#                                 np.random.randint(1, 6, 10), np.random.randint(1, 9, 10),
#                                 np.random.randint(1, 5, 10)])
#
# diagnose_date = np.tile(diagnose_date, 18)
# diagnose_model = np.repeat('Our_model', len(diagnose_date))
#
# test_csv = pd.DataFrame({'image_idx': images_idx, 'image_num': images_num,
#                          'diagnose_date': diagnose_date, 'evaluator': evaluator, 'model_diagnose': model_diagnose, 'model_name': diagnose_model})
#
#
# test_csv.to_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/test.csv',
#                 index=False)

test_csv = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/reviewers_malignant.csv')
ai_malignant = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/AI_model_mel.csv')

ai_diagnose = []

for pred in np.array(ai_malignant['diagnose_res']):
    try:
        ai_diagnose.append(eval(pred).index('malignant')+1)
    except ValueError:
        ai_diagnose.append(-1)    # if plot benign, set this as 0

test_csv['model_name'] = np.repeat('Our_model', 1068)
test_csv['model_diagnose'] = np.repeat(ai_diagnose, 12)
test_csv['diagnose_date'][test_csv['diagnose_date'] == 0] = -1

image_id = np.unique(test_csv['image_id'])
image_num = np.array([np.unique(test_csv['image_num'][test_csv['image_id'] == id]) for id in image_id]).squeeze()
print(image_num)
diagnose_num = np.array(test_csv['diagnose_date'])
diagnose_num[diagnose_num == -1] = 0
diagnose_num[diagnose_num != 0] = 1

diagnose_num = [np.sum(diagnose_num[test_csv['image_id'] == id]) for id in image_id]
diagnose_num = ['{}/12'.format(n) for n in diagnose_num]
plot_res = True

fail_cases = []
for i in range(1, 90):
    res = list(test_csv['diagnose_date'][test_csv['image_id'] == i])
    if -1 in res:
        fail_cases.append(-1)
    else:
        fail_cases.append(0)

if plot_res:
    f, ax = plt.subplots(figsize=(25, 6))
    sns.despine(bottom=True, left=True)
    sns.stripplot(y="diagnose_date", x="image_id", hue='evaluator', dodge=True,
                  data=test_csv, alpha=.85, zorder=0, edgecolor='gray', size=3, marker='o')

    sns.pointplot(y='model_diagnose', x="image_id", data=test_csv, dodge=False, join=False, color="#322f3d", hue='model_name',
                  markers="d", scale=.6)

    # plt.legend(loc=9, bbox_to_anchor=(0., 0.96, 1., .102), borderaxespad=0., ncol=10)
    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:13], labels[:13], handletextpad=0, columnspacing=1, loc=9, bbox_to_anchor=(0., 0.98, 1., .102),
              borderaxespad=0., ncol=13, frameon=True, fontsize=11)

    b = sns.barplot(y=image_num, x=image_id, dodge=False, alpha=.2,  color="gray", saturation=0.5)

    for id in image_id:
        b.text(id-1, image_num[id-1]+0.2, diagnose_num[id-1], color='black', ha="center", fontsize=5.5)

    sns.barplot(y=fail_cases, x=np.arange(1, len(fail_cases)+1), dodge=False, data=test_csv, alpha=.2,
                color="lightcoral", saturation=0.5)
    plt.xticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Number of malignant melanoma')
    plt.ylabel('Length of image sequence')
    plt.tight_layout()
    # plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/diagnose.png', dpi=600)
    plt.show()






