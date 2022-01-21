import os
import torch
import numpy as np
from glob import glob
from _collections import OrderedDict
import matplotlib.pyplot as plt


threshold = 0.51225
test_res = '/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/cnn-diff-hc_MIC/old/with_rank_loss_rank_new_&margin_avg/' \
           'seq_length_4/fold_1/test_res.pt'

mel = ['SMI_Malignant/107_34850141','SMI_Malignant/121_25240566','SMI_Malignant/128_25040662', 'SMI_Malignant/139_14830030',
'SMI_Malignant/14031761', 'SMI_Benign/14270796', 'SMI_Malignant/151_34640257','SMI_Malignant/157_15190897',
       'SMI_Malignant/16_14960021', 'SMI_Malignant/31_25830442',
'SMI_Malignant/35911705', 'SMI_Malignant/36970572','SMI_Malignant/59_26341553',

'SMI_Malignant/63_13450694',

'SMI_Malignant/66_45150184',
'SMI_Malignant/71_14420764',
'SMI_Malignant/80_14950761',
 'SMI_Malignant/92_34650181',
'SMI_Malignant/61_24820521(0)',
'SMI_Malignant/145_25540438(0)']


with open(test_res, 'rb') as f:
    results = torch.load(f)

exit_scores = results['exit_scores']
target = results['target']

total_num = []
for a in mel:
    a = os.path.join('/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned', a)
    total_num.append(len(glob(os.path.join(a, "*MIC*"))))


def pick_correct_date(pred_list):
    assert len(pred_list) == 4

    for i in range(4):
        if pred_list[i] > threshold:
            return i+1
        else:
            print('no large than threshold!!!')
    return 0


diag_res = OrderedDict({'date': [], 'score': []})
for t in range(len(target)):
    if target[t] == 1:
        date = pick_correct_date(exit_scores[t])
        diag_res['date'].append(date)
        diag_res['score'].append(exit_scores[t])
        print(diag_res['date'], diag_res['score'])

date = diag_res['date']
print(total_num)
plt.plot(date)
plt.plot(total_num)
plt.show()
# torch.save(diag_res, '/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/cnn-diff-hc_MIC/old'
#                      '/with_rank_loss_rank_new_&margin_avg/seq_length_4/diagnosis.pt')