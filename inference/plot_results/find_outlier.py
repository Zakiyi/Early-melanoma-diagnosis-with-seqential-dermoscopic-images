import numpy as np
from glob import glob
import os
import torch
import matplotlib.pyplot as plt
res_dirs = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/bce_sat_tkd (copy)'
# for dir in glob(os.path.join(res_dirs, 'r*')):
#     res = torch.load(os.path.join(dir, 'seq_length_4/fold_0/test_res.pt'))
#     target = np.array(res['target'])
#     pred = np.array(res['pred_label'])
#     err_pred = (target == pred).astype(np.int)
#     print(err_pred)


res = torch.load(os.path.join(res_dirs, 'seq_length_4/fold_1/exit_scores_eval.pt'))['time_2']
target = np.array(res['targets'])
pred = np.array(res['pred_labels'])
score = np.array(res['pred_scores'])
err_pred = (target == pred).astype(np.int)
print(err_pred)

plt.scatter(x=np.arange(len(target)), y=target, c=np.array(['C0', 'C1'])[target])
plt.scatter(x=np.arange(len(target)), y=score, c=np.array(['C2', 'C3'])[target])
plt.plot(np.arange(len(target)), np.ones(len(target))*0.53)
plt.show()

