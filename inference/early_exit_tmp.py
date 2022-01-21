import os
import json
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from _collections import OrderedDict
training_result_dirs = '/home/zyi/My_disk/medical_AI/cnn-diff-hc-sa/exp1/pre_trained_clss_4'

Auc = OrderedDict({'time0': [], 'time1': [], 'time2': [], 'time3': []})

for training_result_dir in glob(os.path.join(training_result_dirs, '*')):
    try:
        result_dir = os.path.join(training_result_dir, 'seq_length_4')
        with open(os.path.join(result_dir, 'time_metrics.json'), 'rb') as f:
            metrics = json.load(f)
        Auc['time0'].append(metrics['time_0']['auc'][0])
        Auc['time1'].append(metrics['time_1']['auc'][0])
        Auc['time2'].append(metrics['time_2']['auc'][0])
        Auc['time3'].append(metrics['time_3']['auc'][0])
    except NotADirectoryError:
        pass

value = [np.mean(Auc['time0']), np.mean(Auc['time1']),
       np.mean(Auc['time2']), np.mean(Auc['time3'])]

err = [np.std(Auc['time0']), np.std(Auc['time1']),
       np.std(Auc['time2']), np.std(Auc['time3'])]

with open(os.path.join(training_result_dirs, 'auc_times.json'), 'w') as f:
    json.dump({'auc': value, 'std': err}, f, sort_keys=True)


# e = [64.84291233362441, 68.8669094493398, 71.24336812997804, 70.85843851710725]
# f = [64.72249675848748, 68.67800534518801, 69.60903720462542, 70.79231827683839]
#
# a=[63.09320671588474, 67.23464912280701, 66.7880745415575, 70.44727580640894]
# b=[64.74145299145299, 69.72172607234526, 70.57258341933263, 71.10037773543965]
# c= [63.44176854277474, 69.53314281178058, 71.13141687174195, 71.46651649863723]
# d = [62.09089796512397, 68.7189674790294, 71.19414675451829, 72.48306805853245]
#
# plt.plot(a, 'coral', linestyle='-', linewidth=1.8)
# plt.plot(b, 'coral', linestyle=':', linewidth=1.8)
# plt.plot(c, 'g', linestyle='-', linewidth=1.8)
# plt.plot(d, 'g', linestyle=':', linewidth=1.8)
# plt.legend(['joint_train: loss/len', 'joint_train: loss', 'pre_train: loss/len', 'pre_train: loss'])
# plt.ylim([50, 80])
#
# plt.plot(f, 'coral', linestyle='-', linewidth=1.8)

training_result_dirs = ['/home/zyi/My_disk/medical_AI/cnn-diff-hc-sa/exp5(pretrain + weight_bce)/round_0',
                        '/home/zyi/My_disk/medical_AI/cnn-diff-hc-sa/exp5(pretrain + weight_bce)/round_1',
                        '/home/zyi/My_disk/medical_AI/cnn-diff-hc-sa/exp5(pretrain + weight_bce)/round_2',
                        '/home/zyi/My_disk/medical_AI/cnn-diff-hc-sa/exp5(pretrain + weight_bce)/round_3']
a = []
for dir in training_result_dirs:
    res_dir = os.path.join(dir, 'dst_bce_0.2_0.8', 'seq_length_4','time_metrics.json')
    print(res_dir)
    with open(res_dir, 'rb') as f:
        m = json.load(f)
    res = np.array([m['time_0']['auc'][0], m['time_1']['auc'][0], m['time_2']['auc'][0], m['time_3']['auc'][0]])
    a.append(res)
print(np.mean(np.stack(a, axis=1), axis=1))

a =  [60.93312975, 67.98106692, 69.8416455,  70.93492002]
b =  [63.93413362, 68.65223367, 69.86138399, 70.15101975]
c = [63.77206213, 68.39952386, 69.5109616,  69.68403522]
d =  [64.89809495, 67.71499828, 69.23350632, 70.7875511]
e =  [62.61273766, 67.37683493, 70.02247559, 71.12571032]

a = [60.93312975, 67.98106692, 69.8416455,  70.93492002]
b = [60.12588976, 64.02965074, 68.06809988, 69.96647267]
c = [62.06778317, 67.7865803,  69.52783897, 70.54087365]
d = [59.56726635, 63.79664636, 68.12603778, 69.59867677]
e = [60.01820043, 64.43316117, 68.12110109, 69.40498399]

a = [60.93312975, 67.98106692, 69.8416455,  70.93492002]
b = [62.99805178, 66.69772565, 70.12034529, 70.45963738]
c = [63.65751353, 67.46426067, 69.94870637, 70.82037555]
d = [61.22627163, 64.83859418, 68.4287843,  67.49222285]
e = [59.21681915, 64.78276491, 67.83700153, 68.51943502]

plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.plot(d)
plt.plot(e)
plt.legend(['dst: 0', 'dst: 0.1', 'dst: 0.2', 'dst: 0.3', 'dst: 0.5'])
plt.title('bce(weight) & dst(kl)')