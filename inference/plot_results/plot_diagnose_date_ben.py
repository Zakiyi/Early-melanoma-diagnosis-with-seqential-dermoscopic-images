import os
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
from inference.plot_results import ben_ids

sns.set(style="whitegrid")

# pick benign lesions from all prediciotns
# all_lesions = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/AI_model_alls.csv')
#
# diagres = []
# scores = []
#
# for id in ben_ids:
#     diagres.append(all_lesions['diagnose_res'][all_lesions['lesion_id'] == np.int(id)].iloc[0])
#     scores.append(all_lesions['confidence'][all_lesions['lesion_id'] == np.int(id)].iloc[0])
#
# csv = pd.DataFrame({'lesion_id': ben_ids, 'diagnose_res': diagres, 'scores': scores})
# csv.to_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/AI_model_ben.csv', index=False)

test_csv = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/reviewers_benign.csv')
ai_malignant = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/AI_model_ben.csv')
#
# for i in range(len(test_csv)):
#     print(len(test_csv['diagnose_result'][i].split('-')))
#     test_csv['image_num'][i] = len(test_csv['diagnose_result'][i].split('-'))

ai_ts_ben = np.array(ai_malignant['first_ts'])
ai_ts_ben[ai_ts_ben == 'NAN'] = '-10'
ai_ts_ben_first = [eval(x) for x in ai_ts_ben]

ai_ts_ben = np.array(ai_malignant['second_ts'])
ai_ts_ben[ai_ts_ben == 'NAN'] = '-10'
ai_ts_ben_second = [eval(x) for x in ai_ts_ben]

ai_diagnose = []
for pred in np.array(ai_malignant['diagnose_res']):
    try:
        ai_diagnose.append(eval(pred).index('malignant')+1)
    except ValueError:
        ai_diagnose.append(-1)

# find transit points for benign

test_csv['model_name'] = np.repeat('Our_model', 1080)
test_csv['model_diagnose'] = np.repeat(ai_diagnose, 12)
test_csv['diagnose_date'][test_csv['diagnose_date'] == 0] = -1

image_id = np.unique(test_csv['image_id'])
image_num = [np.unique(test_csv['image_num'][test_csv['image_id'] == id])[0] for id in image_id]
diagnose_num = np.array(test_csv['diagnose_date'])

diagnose_num[diagnose_num != -1] = 0
diagnose_num[diagnose_num == -1] = 1

diagnose_num = [np.sum(diagnose_num[test_csv['image_id'] == id]) for id in image_id]
diagnose_num = ['{}/12'.format(n) for n in diagnose_num]
plot_res = True

correct_cases = []
for i in range(1, 91):
    res = list(test_csv['diagnose_date'][test_csv['image_id'] == i])
    if -1 in res:
        correct_cases.append(-1)
    else:
        correct_cases.append(0)

if plot_res:
    f, ax = plt.subplots(figsize=(26, 6))
    sns.despine(bottom=True, left=True)
    sns.stripplot(y="diagnose_date", x="image_id", hue='evaluator', dodge=True,
                  data=test_csv, alpha=1.0, zorder=0, edgecolor='gray', size=4, marker='o')

    sns.pointplot(y='model_diagnose', x="image_id", data=test_csv, dodge=False, join=False, color="#322f3d", hue='model_name',
                  markers="d", scale=.6)

    sns.pointplot(y=np.array(ai_ts_ben_first), x=np.arange(1, 91), dodge=False, join=False, color="#322f3d",
                  markers="x", scale=.6)

    sns.pointplot(y=np.array(ai_ts_ben_second), x=np.arange(1, 91), dodge=False, join=False, color="#322f3d",
                  markers="x", scale=.6)
    # plt.legend(loc=9, bbox_to_anchor=(0., 0.96, 1., .102), borderaxespad=0., ncol=10)
    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:13], labels[:13], handletextpad=0, columnspacing=1, loc=9, bbox_to_anchor=(0., 0.98, 1., .102),
              borderaxespad=0., ncol=13, frameon=True, fontsize=11)

    b = sns.barplot(y=image_num, x=image_id, dodge=False, alpha=.2,  color="gray", saturation=0.5)

    for id in image_id:
        b.text(id-1, image_num[id-1]+0.2, diagnose_num[id-1], color='black', ha="center", fontsize=5.5)

    sns.barplot(y=correct_cases, x=np.arange(1, len(correct_cases) + 1), dodge=False, data=test_csv, alpha=.2,
                color="lightseagreen", saturation=0.5)

    plt.xticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Number of benign lesions')
    plt.ylabel('Length of image sequence')
    plt.ylim([-1.2, 13])
    plt.tight_layout()
    plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/diagnose_benignstar.png', dpi=600)
    plt.show()






