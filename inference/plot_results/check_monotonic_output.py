import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sn

def get_time_prediction(prediction, type='spatial'):
    spatial_score = prediction['spatial_score']
    temporal_score = prediction['temporal_score']
    exit_score = prediction['exit_score']
    label = prediction['target']

    if type == 'exit':
        score_list = exit_score
    elif type == 'spatial':
        score_list = spatial_score
    elif type == 'temporal':
        score_list = temporal_score
    else:
        raise ValueError

    label = [x for x in label]
    a = score_list[0]
    print(a)
    for i in range(len(score_list)):

        score = [torch.sigmoid(x).cpu().numpy() for x in score_list[i]]

        if label[i] > 0:
            plt.plot(score, 'r:', linewidth=0.8)
        else:
            plt.plot(score, 'b:', linewidth=0.8)

    avg_pred = torch.stack([torch.stack(x) for x in score_list], dim=1)   # Time*N

    label = np.stack(label)
    # print('before ', avg_pred.reshape(-1, 4))
    # print('after ', torch.nn.functional.softmax(avg_pred.reshape(-1, 4), dim=1))

    if type == 'spatial' or type == 'exit':
        # pos = avg_pred[:, label > 0].detach().cpu().numpy()
        # neg = avg_pred[:, label == 0].detach().cpu().numpy()
        pos = torch.sigmoid(avg_pred[:, label > 0]).detach().cpu().numpy()
        neg = torch.sigmoid(avg_pred[:, label == 0]).detach().cpu().numpy()

    if type == 'temporal':
        pos = avg_pred[:, label > 0]
        neg = avg_pred[:, label == 0]
        #
        # pos = pos / pos.norm(p=1, dim=0, keepdim=True)
        # neg = neg / neg.norm(p=1, dim=0, keepdim=True)
        # pos = torch.cumsum(torch.sigmoid(pos), dim=0).detach().cpu().numpy()
        # neg = torch.cumsum(torch.sigmoid(neg), dim=0).detach().cpu().numpy()

        # sep
        # pos = pos.detach().cpu().numpy()
        # neg = neg.detach().cpu().numpy()

        # # sep + sigmoid
        pos = pos.detach().cpu().sigmoid().numpy()
        neg = neg.detach().cpu().sigmoid().numpy()

        # # cumsum
        # pos = torch.cumsum(pos, dim=0).detach().cpu().sigmoid().numpy()
        # neg = torch.cumsum(neg, dim=0).detach().cpu().sigmoid().numpy()

        # cumsum sigmoid
        # pos = torch.cumsum(torch.sigmoid(pos), dim=0).detach().cpu().numpy()/3
        # neg = torch.cumsum(torch.sigmoid(neg), dim=0).detach().cpu().numpy()/3
        # # print('pos ', pos)

        # cumsum
        # pos = torch.cumsum(pos, dim=0).detach().cpu()
        # neg = torch.cumsum(neg, dim=0).detach().cpu()
        #
        # pos = (pos / torch.tensor([1., 2., 3.]).view(3, 1)).sigmoid().numpy()
        # neg = (neg / torch.tensor([1., 2., 3.]).view(3, 1)).sigmoid().numpy()
        # print('posa ', pos)

        # pos = pos).detach().cpu().numpy()
        # neg = torch.sigmoid(neg).detach().cpu().numpy()
    # pos = avg_pred[:, label > 0].detach().cpu().numpy()
    # neg = avg_pred[:, label == 0].detach().cpu().numpy()
    # plt.figure()
    # plt.plot(pos.mean(axis=1), color='rebeccapurple', linewidth=1.5)
    # plt.plot(neg.mean(axis=1), color='sienna', linewidth=1.5)
    # # plt.fill_between([0, 1, 2, 3], pos.mean(axis=1)-pos.std(axis=1), pos.mean(axis=1)+pos.std(axis=1), color='slateblue', alpha=0.18)
    # plt.legend(['melanoma', 'benign'])
    # plt.show()
    return pos, neg


if __name__ == "__main__":
    exp_dir = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/bce (val)/seq_length_4'
    pos = []
    neg = []
    score_type = 'exit'
    sn.set_style('darkgrid')
    for fold in [0, 1, 2, 3, 4]:
        prediction = torch.load(os.path.join(exp_dir, 'fold_{}'.format(fold), 'test_res.pt'))
        tmp_pos, tmp_neg = get_time_prediction(prediction, type=score_type)

        tmp_pos = tmp_pos.mean(axis=1)
        tmp_neg = tmp_neg.mean(axis=1)
        pos.append(tmp_pos)
        neg.append(tmp_neg)

        # pos = np.array(pos).squeeze().transpose(1, 0)  # Nfold * Time
        # neg = np.array(neg).squeeze().transpose(1, 0)

    pos = np.array(pos).squeeze()  # Nfold * Time
    neg = np.array(neg).squeeze()

    plt.figure(figsize=(4.8, 3.2))
    plt.plot(np.arange(pos.shape[-1]), pos.mean(axis=0).squeeze(), color="#2E8B57", linewidth=1.8)
    plt.plot(np.arange(neg.shape[-1]), neg.mean(axis=0).squeeze(), color="#DB7093", linewidth=1.8)

    plt.fill_between(np.arange(pos.shape[-1]), pos.mean(axis=0) - pos.std(axis=0), pos.mean(axis=0) + pos.std(axis=0), color="#2E8B57", alpha=0.15)
    plt.fill_between(np.arange(neg.shape[-1]), neg.mean(axis=0) - neg.std(axis=0), neg.mean(axis=0) + neg.std(axis=0), color="#DB7093", alpha=0.15)
    plt.legend(['malignant', 'benign'], loc='lower left')
    # plt.tight_layout()
    # plt.xlabel('time step', fontsize=12)
    plt.ylabel('averaged prediction scores', fontsize=10)
    # plt.title('Predictions distribution of each time step', fontsize=12)
    plt.xlim([-0.1, 3.2])
    plt.ylim([0.2, 0.75])
    plt.xticks([0, 1, 2, 3], ['Time 1', 'Time 2', 'Time 3', 'Time 4'], fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'pred_{}.png'.format(score_type)), dpi=300)
    plt.show()
    # plt.legend(['CNN-Score-Fusion', 'Ours', 'CNN-LSTM'])
    # plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/ssss/pred.png', dpi=600)
