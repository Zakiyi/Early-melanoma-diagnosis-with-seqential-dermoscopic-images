import os
import numpy as np
import pandas as pd
from glob import glob

# # read diagnose_file
# df = pd.read_excel('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/Full data diagnosis review.xlsx')
#
# # create template
# evaluator = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12']
# experience = ['<5 years', '<5 years', '≥5 years', '<5 years', '≥5 years', '≥5 years', '<5 years', '≥5 years',
#               '≥5 years', '≥5 years', '≥5 years', '<5 years']
# case_list = glob('/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned/SMI_Malignant/*')
# case_idx = []
# image_num = []
# for case in case_list:
#     if '(1)' not in case:
#         name = case.split('/')[-1].split('(')[0]
#         image_num.append(len(glob(os.path.join(case, '*MIC*'))))
#         case_idx.append(name)
#
# # case_idx = np.unique(case_idx)
#
# print('Total number of melanoma is {}'.format(len(case_idx)))
# lesion_id = np.repeat(case_idx, 12)
# image_idx = np.repeat(np.arange(1, 92), 12)
# image_num = np.repeat(image_num, 12)
# evaluator = np.tile(evaluator, len(case_idx))
# experience = np.tile(experience, len(case_idx))
#
#
# diagnose_date = np.arange(len(experience))
# diagnose_res = np.arange(len(experience))
# confidence = np.arange(len(experience))
#
# csv = pd.DataFrame({'lesion_id': lesion_id, 'image_id': image_idx, 'image_num': image_num, 'evaluator': evaluator,
#                     'diagnose_date': diagnose_date, 'diagnose_res': diagnose_res, 'confidence': confidence})
#
# csv.to_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/review_temp1.csv',
#                 index=False)

csv = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/benign_reviewers.csv')
diagnose_res = csv['diagnose_result']
diagnose_date = []
for r in diagnose_res:
    if 'Malignant' not in r and 'Melanoma' not in r:
        diagnose_date.append(0)
    else:
        if 'Melanoma' in r:
            r = r.replace('Melanoma', 'Malignant')
        print(r)
        date = r.split('-').index('Malignant') + 1
        print(len(r.split('-')), date)
        diagnose_date.append(date)

csv['diagnose_date'] = diagnose_date
csv.to_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/reviewers_benign.csv',
                index=False)