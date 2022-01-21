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

res_dir = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/' \
          'CST features/SCA/seq_length_4'

# collect predicted scores and labels form each fold
for f in [0, 1, 2, 3, 4]:
    predictions = torch.load(os.path.join(res_dir, 'fold_{}/diagnose_res.pt'.format(f)))
    evals = torch.load(os.path.join(res_dir, 'fold_{}/exit_scores_eval.pt'.format(f)))['time_3']

    scores = evals['pred_scores']
    target = evals['targets']

    case_list = split[f]['val']
    assert len(case_list) == len(predictions)
    assert len(case_list) == len(scores)

    for i in range(len(case_list)):
        assert len(predictions[i]) == len(glob(os.path.join(data_root, case_list[i], "*MIC*")))
        case_id = case_list[i].split('/')[-1].split('_')[-1].split('(')[0]
        pred = predictions[i]
        score = scores[i]
        lesions.append(case_id)
        diagnose_res.append(pred)
        confidence.append(score)
        targets.append(target[i])

csv_model = pd.DataFrame({'lesion_id': lesions, 'diagnose_res': diagnose_res, 'confidence': confidence,
                    'target': targets})

csv_model.to_csv(os.path.join(res_dir, 'AI_model.csv'), index=False)

"""only add malignant lesions"""
from inference.plot_results import mel_ids

diagnose_res = []
confidence_score = []
lesion_id = []

for id in mel_ids:
    lesion_id.append(id)
    print(id)
    print(csv_model['diagnose_res'][csv_model['lesion_id'] == str(id)])
    diagnose_res.append(csv_model['diagnose_res'][csv_model['lesion_id'] == str(id)].iloc[0])
    confidence_score.append(csv_model['confidence'][csv_model['lesion_id'] == str(id)].iloc[0])

csv = pd.DataFrame({'lesion_id': lesion_id, 'diagnose_res': diagnose_res, 'confidence_score': confidence_score})
csv.to_csv(os.path.join(res_dir, 'AI_model_mel.csv'), index=False)

