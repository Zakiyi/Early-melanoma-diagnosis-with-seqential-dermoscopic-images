import os
import torch
from glob import glob
import pickle as pkl
import pandas as pd
import numpy as np


with open('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/data_setting/data_split_replaced.pkl', 'rb') as f:
    split = pkl.load(f)

data_root = '/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned_replaced'
lesions = []
diagnose_res = []
confidence = []
targets = []
for f in [0, 1, 2, 3, 4]:
    predictions = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
                             'CST features/SCA/seq_length_4/fold_{}/diagnose_res.pt'.format(f))

    evals = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
                       'CST features/SCA/seq_length_4/fold_{}/exit_scores_eval.pt'.format(f))['time_3']
    scores = evals['pred_scores']
    target = evals['targets']

    case_list = split[f]['val']
    assert len(case_list) == len(predictions)
    assert len(case_list) == len(scores)
    for i in range(len(case_list)):
        assert len(predictions[i]) == len(glob(os.path.join(data_root, case_list[i], "*MIC*")))
        case_id = case_list[i].split('/')[-1]
        pred = predictions[i]
        score = scores[i]
        lesions.append(case_id)
        diagnose_res.append(pred)
        confidence.append(score)
        targets.append(target[i])

csv = pd.DataFrame({'lesion_id': lesions, 'diagnose_res': diagnose_res, 'confidence': confidence,
                    'target': targets})

# csv.to_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/compares.csv', index=False)
#

"""only add malignant lesions"""
csv_model = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
                        'human_results/compares.csv')

csv_human = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
                        'human_results/review_temp.csv')

# 165_45070176 is wrong malignant
case_id = []
for i in range(1, 90):
    id = np.unique(csv_human['lesion_id'][csv_human['image_id'] == i])
    case_id.append(id[0])

diagnose_res = []
confidence_score = []
lesion_id = []

for id in case_id:
    lesion_id.append(id)
    print(id)
    print(csv_model['diagnose_res'][csv_model['lesion_id'] == id])
    diagnose_res.append(csv_model['diagnose_res'][csv_model['lesion_id'] == id].iloc[0])
    confidence_score.append(csv_model['confidence'][csv_model['lesion_id'] == id].iloc[0])

csv = pd.DataFrame({'lesion_id': lesion_id, 'diagnose_res': diagnose_res, 'confidence_score': confidence_score})
csv.to_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
           'human_results/AI_model.csv', index=False)

