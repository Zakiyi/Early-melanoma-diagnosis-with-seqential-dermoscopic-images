from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import matplotlib
import seaborn
classes = ['benign', 'malignant']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']


def tsne_transform(features, labels, score=None, interval=None):
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=8)
    Labels = []

    for x in labels[:]:
        if x == 0:
            Labels.append('benign')
        else:
            Labels.append('malignant')
    # score = np.concatenate([np.ones(37), 2*np.ones(37), 4*np.ones(37), 8*np.ones(37)])

    tsne_embeddings = tsne.fit_transform(features)

    if interval is not None:
        tsne_embeddings = pd.DataFrame({'label': Labels,
                                        'dim1': tsne_embeddings[interval[0]:interval[1], 0],
                                        'dim2': tsne_embeddings[interval[0]:interval[1], 1]})
    else:
        tsne_embeddings = pd.DataFrame({'label': Labels,
                                        'dim1': tsne_embeddings[:, 0],
                                        'dim2': tsne_embeddings[:, 1]})
    return tsne_embeddings


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    # plt.figure(figsize=(10, 10))
    print(targets)
    for i in range(2):
        inds = np.where(targets == i)
        print('inds ', inds)
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.show(block=1)


if __name__ == '__main__':
    matplotlib.style.use('seaborn')
    # load baseline results
    res = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/bce/seq_length_4/fold_2/spatiotemporal_feat.pt')
    # res = torch.load(
    #     '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/dst_bce_0.2_0.8 (val)/seq_length_4/fold_2/spatiotemporal_feat.pt')

    # score = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/bce_sat_tkd/seq_length_4/fold_4/exit_scores_eval.pt')
    # score = score['time_0']['pred_scores']
    # features = res['att_features'][:, 3, :]
    target = np.array(res['target'])
    print(len(target))
    features = np.concatenate([res['att_features'][:, 0, :],
                               res['att_features'][:, 1, :],
                               res['att_features'][:, 2, :],
                               res['att_features'][:, 3, :]],
                              axis=0)

    target = np.concatenate([np.array(res['target']), np.array(res['target']),
                             np.array(res['target']), np.array(res['target'])])

    tsne_embeddings = tsne_transform(features, target)
    f = sns.lmplot(x='dim1', y='dim2', data=tsne_embeddings, fit_reg=False, legend=False, palette=dict(benign="#2ca02c", malignant="palevioletred"),
               size=3, hue='label', scatter_kws={"s": 60, "alpha": 0.5})
    f.set_ylabels('')
    f.set_xlabels('time 0')
    f.set_xticklabels([])
    f.set_yticklabels([])
    # seaborn.despine(left=False)

    # ax = sns.scatterplot(x="dim1", y="dim2", legend=False, alpha=0.5, palette=dict(benign="#2ca02c", malignant="#e377c2"),
    #                      hue="label", data=tsne_embeddings)
    # plt.axis('off')
    # plt.tight_layout()
    # load sat&tkd results
    # res = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/bce_sat_tkd/seq_length_4/fold_1/spatiotemporal_feat.pt')
    # target = np.array(res['target'])
    # features = res['att_features'][:, 3, :]
    # tsne_embeddings = tsne_transform(features, target)
    # g = sns.lmplot(x='dim1', y='dim2', data=tsne_embeddings, fit_reg=False, legend=False, palette=dict(benign="#2ca02c", malignant="palevioletred"),
    #            size=3, hue='label', scatter_kws={"s": 50, "alpha": 0.5})
    # plt.axis('off')
    # plt.tight_layout()
    # # ax = sns.scatterplot(x="dim1", y="dim2", legend=False, alpha= 0.5, palette=dict(benign="#2ca02c", malignant="#e377c2"),
    # #                      hue="label", size='pred', data=tsne_embeddings)
    # plt.axis('off')

    plt.show()
    # test_res = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/bce_sat_tkd/seq_length_4'
    # folders = glob(os.path.join(test_res, 'fold*'))
    # target = []
    # features = []
    # for i in range(len(folders)):
    #     res = torch.load(os.path.join(folders[i], 'spatiotemporal_feat.pt'))
    #     target.append(np.array(res['target']))
    #     features.append(res['att_features'][:, 1, :]+(i*2))
    #
    # target = np.concatenate(target)
    # features = np.concatenate(features, axis=0)
    # tsne_embeddings = tsne_transform(features, target)
    # g = sns.lmplot(x='dim1', y='dim2', data=tsne_embeddings, fit_reg=False, legend=True, palette="muted",
    #            size=4, hue='Label', scatter_kws={"s": 50, "alpha": 0.5})
    # plt.show()
    #

    fig, axs = plt.subplots(ncols=4, figsize=(10, 2.4))
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.90, top=0.98, wspace=0.02, hspace=None)
    features = np.concatenate([res['att_features'][:, 0, :],
                               res['att_features'][:, 1, :],
                               res['att_features'][:, 2, :],
                               res['att_features'][:, 3, :]],
                              axis=0)

    for i in range(4):
        # features = res['att_features'][:, i, :]
        target = np.array(res['target'])
        # tsne_embeddings = tsne_transform(features, target)
        tsne_embeddings = tsne_transform(features, target, interval=[i*159, (i+1)*159])
        if i < 3:
            sns.scatterplot(x="dim1", y="dim2", legend=False, alpha=0.5, palette=dict(benign="#2ca02c", malignant="#e377c2"),
                    hue="label", data=tsne_embeddings, ax=axs[i])
        else:
            sns.scatterplot(x="dim1", y="dim2", legend='brief', alpha=0.5,
                            palette=dict(benign="#2ca02c", malignant="#e377c2"),
                            hue="label", data=tsne_embeddings, ax=axs[i])
            # box = axs[i].get_position()  # get position of figure
            # axs[i].set_position([box.x0, box.y0, box.width, box.height])  # resize position

            # Put a legend to the right side
            axs[i].legend(loc='center right', bbox_to_anchor=(1.48, 0.5), ncol=1, prop={'size': 9})

        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        # axs[i].set_xlabel('time {}'.format(i), fontsize=12)
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
    plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/compares/'
                'ablation_results/bce_fold_2_train.png', dpi=300)
    # plt.tight_layout()
    plt.show()


