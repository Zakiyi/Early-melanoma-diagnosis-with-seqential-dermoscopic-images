import pandas as pd
import numpy as np

csv_mel = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
                      'human_results/reviewers_malignant.csv')

diag_res = csv_mel['diagnose_result']
Reviewers = np.unique(csv_mel['evaluator'])
acc = []
for r in Reviewers:
    res = diag_res[csv_mel['evaluator'] == r]
    first_pred = [p.split('-')[0] for p in res]
    assert len(np.unique(first_pred)) == 2
    print(r, first_pred.count('Malignant'))


