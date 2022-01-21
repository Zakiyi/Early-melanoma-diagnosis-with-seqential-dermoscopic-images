import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn')
# sns.set_color_codes("dark")

our = [0.5652, 0.7843, 0.8478, 0.9168]
our_baseline = [0.5265, 0.8204, 0.6860, 0.7530]


lstm = [0.7171, 0.8682, 0.7202, 0.8465]
singlecnn = [0.32522196, 0.42522985, 0.6295706,  0.65806323]

plt.figure(figsize=(6, 3.5))

plt.plot(singlecnn, '-o', color='#4C72B0')
plt.plot(lstm, '-o', color='#55A868')
plt.plot(our_baseline, '-o', color='#C44E52')
plt.plot(our, '-o', color='#8172B2')
# plt.plot(our_baseline, '-o', color='#4C72B0')
# plt.plot(our, '-o', color= '#C44E52')

plt.legend(['Single-img-CNN(HAM)', 'CNN-LSTM', 'CST-Baseline', 'CST-SCA-TKD'])
# plt.legend(['CST-Baseline', 'CST-SCA-TKD'])
plt.ylabel('prediction scores', fontsize=10)
# plt.ylim([0.5, 0.9])
plt.xticks([0, 1, 2, 3], ['Time 1', 'Time 2', 'Time 3', 'Time 4'], fontsize=10)
plt.tight_layout()
plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/individual_lesion/malignant/13971557/plot.png', dpi=300)
plt.show()