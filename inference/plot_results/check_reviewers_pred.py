import os
import pandas as pd
import numpy as np
from _collections import OrderedDict

csv2check = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/reviewers_R12.csv'
ZY_csv = pd.read_csv(csv2check)

JNF_csv = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/diffcult.csv')
JNF_res = np.array(JNF_csv['12R'])  # res from jennifer

index = [ZY_csv.index[ZY_csv['id'] == id].tolist()[0] for id in JNF_csv['ID']]

ZY_pred = np.array(ZY_csv['pred'][index])
ZY_pred[ZY_pred == 'Malignant'] = 1
ZY_pred[ZY_pred == 'Melanoma'] = 1
ZY_pred[ZY_pred == 'Benign'] = 0

labels = np.array(ZY_csv['label'][index])
ids = np.array(ZY_csv['id'][index])
ZY_res = np.array([ZY_pred[i] == labels[i] for i in range(len(ZY_pred))]).astype(np.int)

err_index = np.where(ZY_res != JNF_res)
print(ids[err_index])
# [25080775 48460554 25120713 14760474 15410724 25410461 46950470 46811354
#  15471241 14310778 15070020 23240499 34650181 14950761 14540425]


def get_pred_cross_time(preds, padding=True):
    time_1_pred = [p.split('-')[0] for p in preds]
    time_2_pred = []
    time_3_pred = []
    time_4_pred = []

    end_pred = [p.split('-')[-1] for p in preds]

    for p in preds:
        p = p.split('-')
        index = np.linspace(0, len(p)-1, 4, endpoint=True, dtype=np.int)
        # print(index)
        if padding:
            time_2_pred.append(p[index[1]])
            time_3_pred.append(p[index[2]])
            time_4_pred.append(p[index[3]])
        else:
            index = index[index != 0]
            if len(p) >= 2:
                time_2_pred.append(p[index[0]])

            if len(p) >= 3:
                time_3_pred.append(p[index[1]])

            if len(p) >= 4:
                time_4_pred.append(p[index[2]])

    return time_1_pred, time_2_pred, time_3_pred, time_4_pred, end_pred
#
#
# reviewers_mel = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/reviewers_malignant (copy).csv')
#
# reviewers_pred = OrderedDict({'R1': [], 'R2': [], 'R3': [], 'R4': [], 'R5': [], 'R6': [], 'R7': [], 'R8': [],
#                               'R9': [], 'R10': [], 'R11': [], 'R12': []})
#
# mel_predictions = reviewers_mel['diagnose_result']
#
# for r, _ in reviewers_pred.items():
#     mel_pred = np.array(mel_predictions[reviewers_mel['evaluator'] == r])
#     # print(r, mel_pred)
#     time_1_mel_pred, time_2_mel_pred, time_3_mel_pred, time_4_mel_pred, end_pred = get_pred_cross_time(mel_pred)
#     print(np.array_equal(np.array(time_4_mel_pred), np.array(end_pred)))
#     if r == 'R1':
#         print(time_4_mel_pred)
