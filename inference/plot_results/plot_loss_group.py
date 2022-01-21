import os
import torch
import matplotlib
import seaborn as sn
import matplotlib.pyplot as plt

matplotlib.style.use('seaborn')
res_dir='/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/TSA/sep_train_batchmean/tkd_dst_klv_avg/round_8'
# 0.1 0.9
a = torch.load(os.path.join(res_dir, 'dst_bce_0.1_0.9/seq_length_4/fold_3/loss_group.pt'))['val']['distill_loss']
# 0.2 0.8
b = torch.load(os.path.join(res_dir, 'dst_bce_0.2_0.8/seq_length_4/fold_3/loss_group.pt'))['val']['distill_loss']
# 0.3 0.7
c = torch.load(os.path.join(res_dir, 'dst_bce_0.3_0.7/seq_length_4/fold_3/loss_group.pt'))['val']['distill_loss']
# 0.4, 0.6
d = torch.load(os.path.join(res_dir, 'dst_bce_0.4_0.6/seq_length_4/fold_3/loss_group.pt'))['val']['distill_loss']
# 0.4, 0.6
e = torch.load(os.path.join(res_dir, 'dst_bce_0.5_0.5/seq_length_4/fold_3/loss_group.pt'))['val']['distill_loss']
# 0.4, 0.6
# f = torch.load(os.path.join(res_dir, 'dst_bce_0.6_0.4/seq_length_4/fold_3/loss_group.pt'))['val']['distill_loss']
f = torch.load(os.path.join(res_dir, 'dst_bce_0_1/seq_length_4/fold_3/loss_group.pt'))['val']['distill_loss']
plt.figure(figsize=(6, 3.4))
plt.plot(f)
plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.plot(d)
plt.plot(e)
plt.ylim([0., 0.35])
plt.ylabel('temporal distillation loss')
plt.xlabel('training epoch')
plt.tight_layout()
plt.legend([r'$\alpha=0$', r'$\alpha=0.1$', r'$\alpha=0.2$', r'$\alpha=0.3$', r'$\alpha=0.4$', r'$\alpha=0.5$'])
# plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/ablation_results/dst_loss.png', dpi=300)
plt.show()

