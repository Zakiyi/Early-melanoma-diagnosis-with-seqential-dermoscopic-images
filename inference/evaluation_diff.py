import os
import cv2
import time
import json
import yaml
import torch
import seaborn as sns
from scipy import interp
import numpy as np
import pandas as pd
from glob import glob
import pickle as pkl
from models import get_model
from collections import OrderedDict
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from data_proc.ssd_dataset import Skin_Dataset
from data_proc.sequence_aug_diff import Augmentations_diff
from models import get_model_output
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
from models import convert_state_dict
from torchvision.utils import make_grid
threshold = 0.6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ['benign', 'malignant']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

img_visualization = []
dif_visualization = []


def hook_forward_img(module, input, output):
    img_visualization.append(output)


def hook_forward_dyn(module, input, output):
    dif_visualization.append(output)


def set_vis_layers(model):
    model.img_sbn_classifier.layer2.register_forward_hook(hook_forward_img)
    model.dym_dfn_classifier.layer2.register_forward_hook(hook_forward_dyn)


def collect_embeddings(features_list, seq_length):
    # seq_length = seq_length-1
    assert len(features_list) % seq_length == 0

    features_list = [x.mean(dim=0) for x in features_list]

    embeddings = []
    for i in range(len(features_list) // seq_length):
        embeddings.append(torch.cat(features_list[i*seq_length:(i+1)*seq_length]))
        # embeddings.append(features_list[i * seq_length + 2])
    embeddings = torch.stack(embeddings, dim=0)
    return embeddings.detach().cpu().numpy()


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


def show_seq_imgs(sequence, pred_score, target):
    # fig, axes = plt.subplots(1, seq_length)
    sequence = sequence['images']
    sequence = sequence[9, ...]

    grid_image = make_grid(sequence).cpu().numpy()
    grid_image = grid_image.transpose(1, 2, 0)
    cv2.normalize(grid_image, grid_image, 0, 255, cv2.NORM_MINMAX)
    plt.imshow(grid_image)
    plt.title('pred_score: {:.4f}'.format(pred_score) + '   ' +'target: {}'.format(target))
    plt.show(block=1)

    # for i in range(seq_length):
    #     img = sequence[i, ...].numpy().transpose(1, 2, 0) * 255
    #     print('max img: ', np.max(img), 'min img: ', np.min(img))
    #     cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    #     axes[i].imshow(img.astype(np.uint8))
    #
    # plt.show(block=1)


def evaluation(cfg, model, data_split_file):

    test_aug_parameters = OrderedDict({'affine': None,
                                       'flip': True,
                                       'color_trans': {'brightness': (1.0, 1.0),
                                                       'contrast': (1.0, 1.0),
                                                       'saturation': (1.0, 1.0),
                                                       'hue': (-0.001, 0.001)},
                                       'normalization': {'mean': (0.485, 0.456, 0.406),
                                                         'std': (0.229, 0.224, 0.225)},
                                       'size': 320,
                                       'scale': (1.0, 1.0),
                                       'ratio': (1.0, 1.0)})
    # (0.763038, 0.54564667, 0.57004464), (0.14092727, 0.15261286, 0.1699712)
    # test_augmentor = Augmentations(test_aug_parameters, test_mode=False)
    test_augmentor = Augmentations_diff(test_aug_parameters, test_mode=True, color_recalibration=True, test_aug='ten_crops', padding_mode='normal')
    dataset = Skin_Dataset(cfg['data']['root'], data_split_file[cfg['fold']]['val'], seq_length=cfg['data']['seq_length'],
                           transform=test_augmentor.transform, data_modality=cfg['data']['modality'], is_train=False,
                           test_mode=True)

    test_results = OrderedDict({'target': [], 'pred_score': [], 'pred_label': [],
                                'spatial_score': [], 'temporal_score': [], 'exit_score': []})
    model.eval()
    '''for extracting the intermediate layers output'''
    # img_visualization = []
    # dif_visualization = []
    # features = np.zeros(len(dataset), 32*cfg['data']['seq_length'])
    # set_vis_layers(model)

    with torch.no_grad():
        for data in iter(dataset):
            outputs, s, t, e = get_model_output(model, data, cfg['data']['modality'], p_index=cfg['data']['p_index'])
            # print('spatial: ', torch.sigmoid(s[0].squeeze().mean()))
            # print('spatial: ', torch.sigmoid(s[1].squeeze().mean()))
            # print('spatial: ', torch.sigmoid(s[2].squeeze().mean()))
            # print('spatial: ', torch.sigmoid(s[3].squeeze().mean()))
            #
            # print('temporal: ', torch.sigmoid(t[0].squeeze().mean()))
            # print('temporal: ', torch.sigmoid(t[1].squeeze().mean()))
            # print('temporal: ', torch.sigmoid(t[2].squeeze().mean()))

            # print('exit: ', torch.sigmoid(e[0].squeeze().mean()))
            # print('exit: ', torch.sigmoid(e[1].squeeze().mean()))
            # print('exit: ', torch.sigmoid(e[2].squeeze().mean()))
            # print('exit: ', torch.sigmoid(e[3].squeeze().mean()))
            s = [x.squeeze().mean() for x in s]
            t = [x.squeeze().mean() for x in t]
            e = [x.squeeze().mean() for x in e]

            test_results['spatial_score'].append(s)
            test_results['temporal_score'].append(t)
            test_results['exit_score'].append(e)

            if isinstance(outputs, tuple) or isinstance(outputs, list):
                outputs = outputs[-1]

            pred_score = outputs.detach().cpu().numpy()  # (Ncrops * seq_length, 1)
            pred_score = np.mean(pred_score, axis=0)

            print('pred_score: ', pred_score)
            print('target: ', data['target'].detach().cpu().numpy())

            test_results['target'] += [data['target'].detach().cpu().numpy()]
            test_results['pred_score'] += [pred_score.squeeze()]
            # test_results['pred_label'] += [get_pred_class_index(pred_score, threshold)]

            # if test_results['target'][-1] != test_results['pred_label'][-1]:
            #     show_seq_imgs(data['image'][cfg['data']['modality']], pred_score, test_results['target'][-1])

    # embeddings = collect_embeddings(img_visualization, cfg['data']['seq_length'])

    fpr, tpr, thresholds = roc_curve(test_results['target'], test_results['pred_score'])
    # precision_fold, recall_fold, thresh = precision_recall_curve(test_results['target'], test_results['pred_score'])
    # precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
    """choose the best threshold"""
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    roc_t_threshold = list(roc_t['threshold'])[0]

    # roc = pd.DataFrame({'tf': pd.Series(tpr - fpr, index=i), 'threshold': pd.Series(thresholds, index=i)})
    # roc_t = roc.iloc[roc.tf.argsort()[-1:]]
    # roc_t_threshold = list(roc_t['threshold'])[0]
    test_results['pred_label'] = list(map(lambda x: 1 if x > roc_t_threshold else 0,  test_results['pred_score']))

    accuracy = metrics.accuracy_score(test_results['target'], test_results['pred_label'])
    auc = metrics.roc_auc_score(test_results['target'], test_results['pred_score'])
    recall = metrics.recall_score(test_results['target'], test_results['pred_label'])
    precision = metrics.precision_score(test_results['target'], test_results['pred_label'])
    f1_score = metrics.f1_score(test_results['target'], test_results['pred_label'])

    tn, fp, fn, tp = confusion_matrix(test_results['target'], test_results['pred_label']).ravel()
    specificity = tn / (tn + fp)
    # return {'accuracy': accuracy, 'auc': auc, 'recall': recall, 'precision': precision, 'f1-score': f1_score}, \
    #        (fpr, tpr), {'embedding': embeddings, 'target': list(np.stack(test_results['target']))}

    return {'accuracy': accuracy, 'auc': auc, 'recall': recall, 'precision': precision,
            'f1-score': f1_score, 'specificity': specificity, 'threshold': roc_t_threshold}, (fpr, tpr), test_results


def cross_fold_validation(cfg, result_dir, best_model=True, show_roc=True):
    folds = glob(os.path.join(result_dir, 'fold*'))
    folds = sorted(folds, key=lambda x: x.split('/')[-1].split('_')[-1])

    model_file = cfg['model'] + '_best.model' if best_model else cfg['model'] + '_final.model'
    cfg['data']['seq_length'] = int(result_dir.split('/')[-1].split('_')[-1])
    print('current sequence length for evaluation is {}'.format(cfg['data']['seq_length']))

    # load data_split file
    run_exp_dir = cfg['run_exp']
    with open(os.path.join(run_exp_dir, 'data_setting', 'data_split.pkl'), 'rb') as f:
        data_split_file = pkl.load(f)

    # run five fold validation
    result_metrics = OrderedDict()
    folds_tpr = []
    base_fpr = np.linspace(0, 1, 101)

    # precision_array = []
    # threshold_array = []
    # recall_array = np.linspace(0, 1, 101)

    color = ['b', 'g', 'r', 'k', 'y']
    for i in range(len(folds)):
        start_time = time.time()
        cfg['fold'] = int(folds[i].split('_')[-1])
        # load model
        model = get_model(cfg['model'], cfg['data']['channel'], 1, cfg['data']['seq_length']).to(device)
        saved_model = torch.load(os.path.join(folds[i], model_file))

        try:
            model.load_state_dict(saved_model['model_state_dict'])  # maybe need to change
        except RuntimeError:
            model_state_dict = convert_state_dict(saved_model['model_state_dict'])
            model.load_state_dict(model_state_dict)

        # evaluation
        print(os.path.join(folds[i], model_file))

        result_metrics['fold_{}'.format(cfg['fold'])], (fpr, tpr), test_res = evaluation(cfg, model, data_split_file)
        torch.save(test_res, os.path.join(folds[i], 'test_res.pt'))
        # global img_visualization
        # global dif_visualization
        # img_visualization = []
        # dif_visualization = []
        #
        # tsne = TSNE(n_components=2, init='pca', random_state=0)
        # tsne_embeddings = tsne.fit_transform(embeddings['embedding'])
        # tsne_embeddings = pd.DataFrame({'Label': embeddings['target'],
        #                                 'dim1': tsne_embeddings[:, 0],
        #                                 'dim2': tsne_embeddings[:, 1]})
        # print(tsne_embeddings)
        # sns.lmplot(x='dim1', y='dim2', data=tsne_embeddings, fit_reg=False, legend=True,
        #            size=9, hue='Label', scatter_kws={"s": 200, "alpha": 0.3})
        # plt.show(block=1)

        plt.plot(fpr, tpr, color[i], alpha=0.1)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        folds_tpr.append(tpr)
        print('{}-th fold evaluated, using time {:.3f}'.format(i, time.time() - start_time))

    np.save(os.path.join(result_dir, 'roc_plot_' + model_file.split('_')[-1].split('.')[0]), folds_tpr)
    with open(os.path.join(result_dir, model_file.split('.')[0] + '_evaluation.json'), 'w') as f:
        json.dump(result_metrics, f, indent=4, sort_keys=True)

    if show_roc:
        folds_tpr = np.array(folds_tpr)
        mean_tprs = folds_tpr.mean(axis=0)
        std = folds_tpr.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        plt.plot(base_fpr, mean_tprs, 'b')
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.show(block=1)


if __name__ == "__main__":
    with open('/home/zyi/MedicalAI/Skin_lesion_prognosis/configs/skin_config.yml', 'r') as f:
        cfg = yaml.load(f)

    training_result_dirs = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/run_exp/cnn-diff-hc_MIC/orginal_random_seq/without_alignment_aligned'
    cfg['model'] = 'cnn-diff-hc'
    seq_length = [4]
    # for training_result_dir in glob(os.path.join(training_result_dirs, '*')):
    #     for length in seq_length:
    #         result_dir = os.path.join(training_result_dir, 'seq_length_{}'.format(length))
    #         cross_fold_validation(cfg, result_dir, best_model=True, show_roc=True)

    for length in seq_length:
        result_dir = os.path.join(training_result_dirs, 'seq_length_{}'.format(length))
        cross_fold_validation(cfg, result_dir, best_model=True, show_roc=True)