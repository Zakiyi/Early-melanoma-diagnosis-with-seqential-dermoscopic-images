import os
import torch
import numpy as np
import pickle as pkl
from _collections import OrderedDict

# test_res = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/CNN_features/single_img_cnn_ham'
#
# for fold in [0, 1, 2, 3, 4]:
#     save_res = OrderedDict({'pred_score': []})
#     for length in [1, 2, 3, 4]:
#         res = torch.load(os.path.join(test_res, 'seq_length_{}'.format(length), 'fold_{}'.format(fold), 'test_res.pt'))
#         save_res['pred_score'].append(np.array(res['pred_score']))
#         target = np.array(res['target'])
#
#     save_res['pred_score'] = np.stack(save_res['pred_score'], axis=1)
#     save_res['target'] = target
#     print(os.path.join(test_res, 'fold_{}.pt'.format(fold)))
#     torch.save(save_res, os.path.join(test_res, 'fold_{}.pt'.format(fold)))

with open('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/data_setting/data_split.pkl', 'rb') as f:
    data = pkl.load(f)

data = data[0]['val']

# CST_SCA_TKD
preds = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/dst_bce_0.2_0.8 (val)/seq_length_4/fold_0/test_res.pt',
                   map_location='cpu')
target = np.array(preds['target'])
pred_our = np.array(preds['exit_score'], dtype=np.float)
pred_our = torch.sigmoid(torch.from_numpy(pred_our))

for i in range(len(target)):
    print(pred_our[i, :].data, target[i], data[i])
# print(target)
# print(pred_our)


# single_img_cnn ham
# preds = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/CNN_features/single_img_cnn_ham/fold_4.pt')
# target = np.array(preds['target'])
# pred_our = preds['pred_score']
# for i in range(len(target)):
#     print(pred_our[i, :], target[i], data[i])
# import os
# from glob import glob
#
# m = glob('/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned/SMI_Benign/*')
# m_o = glob('/home/zyi/MedicalAI/Original skin data/aaaaa/SMI_Benign/*')
#
# m = [os.path.basename(x).split('(')[0] for x in m]
# m_o = [os.path.basename(x).split('(')[0] for x in m_o]
#
# for x in m:
#     if '0' not in m_o:
#         print(x)
#
