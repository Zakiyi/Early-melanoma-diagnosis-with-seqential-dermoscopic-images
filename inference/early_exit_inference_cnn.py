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
from data_proc.sequence_aug import Augmentations
from models import get_model_output
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from models import convert_state_dict
from sklearn.manifold import TSNE
from data_proc.ssd_dataset_test import Skin_Dataset_test
threshold = 0.6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ['benign', 'malignant']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

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

"""
only for single-img-cnn
that is we train a cnn on sequential images without considering the temporal relations
and during interference, we directly compute score at each time point for a sequential inputs
"""

def tsne_transform(features, labels):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_embeddings = tsne.fit_transform(features)
    tsne_embeddings = pd.DataFrame({'Label': labels,
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


def get_model_output(model, data):
    """

    :param model:
    :param data: """ """
    :return: ({'seq_0': {'avg_score': [], 'spatial': [], 'temporal': [], 'exit': []},
               'seq_1': {'avg_score': [], 'spatial': [], 'temporal': [], 'exit': []},
               'seq_2': {'avg_score': [], 'spatial': [], 'temporal': [], 'exit': []}
               })
    """
    assert isinstance(data['image'], OrderedDict)
    outputs = OrderedDict()
    print('data seq num: {}'.format(len(data['image'])))

    for name, img_seq in data['image'].items():
        preds = OrderedDict({})
        exit_scores = []
        for i in range(img_seq.shape[1]):
            avg_preds = model(img_seq[:, i, ...].unsqueeze(dim=1).to(device))
            exit_scores.append(avg_preds)

            if isinstance(avg_preds, tuple) or isinstance(avg_preds, list):
                avg_preds = avg_preds[-1].detach().cpu().numpy().mean(axis=0).squeeze()
                exit_scores.append(avg_preds)

        exit_scores = [x.squeeze().detach().cpu().mean().sigmoid().numpy() for x in exit_scores]

        preds['exit_score'] = exit_scores

        outputs[name] = preds
    return outputs


def compute_single_exit_score_performance(pred_score, target):
    fpr, tpr, thresholds = roc_curve(target, pred_score)

    """choose the best threshold"""
    # https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    # gmeans = np.sqrt(tpr * (1 - fpr))
    # ix = np.argmax(gmeans)
    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # roc_t_threshold = thresholds[ix]

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    roc_t_threshold = list(roc_t['threshold'])[0]
    pred_label = list(map(lambda x: 1 if x > roc_t_threshold else 0, pred_score))

    accuracy = metrics.accuracy_score(target, pred_label)
    auc = metrics.roc_auc_score(target, pred_score)
    recall = metrics.recall_score(target, pred_label)
    precision = metrics.precision_score(target, pred_label)
    f1_score = metrics.f1_score(target, pred_label)

    tn, fp, fn, tp = confusion_matrix(target, pred_label).ravel()
    specificity = tn / (tn + fp)

    return {'accuracy': accuracy, 'auc': auc, 'recall': recall, 'precision': precision,
            'f1-score': f1_score, 'specificity': specificity}, (fpr, tpr), roc_t_threshold, pred_label


def evaluation(cfg, model, data_split_file):
    """

    :param cfg:
    :param model:
    :param data_split_file:
    :return:
    """

    # when setting test_mode as true, only 'normalization' in test_aug_parameters is used
    test_augmentor = Augmentations(test_aug_parameters, test_mode=True)

    # evaluate auc at each exit step: e.g.(0, 2, 4, 7); and compute the optimal threshold
    dataset = Skin_Dataset_test(cfg['data']['root'], data_split_file[cfg['fold']]['val'],
                                seq_length=cfg['data']['seq_length'], transform=test_augmentor.transform, seq_type='all')

    test_results = OrderedDict({'target': [], 'avg_score': [], 'pred_label': [], 'features': [], 'att_features': [], 'exit_score': []})
    model.eval()

    with torch.no_grad():
        for data in iter(dataset):
            outputs = get_model_output(model, data)

            test_results['exit_score'].append(outputs['seq_0']['exit_score'])
            test_results['target'] += [data['target'].detach().cpu().numpy()]

    test_results['exit_score'] = np.stack(test_results['exit_score'], axis=0)  # N*time_step

    print('exit score length: ', test_results['exit_score'].shape[-1])

    # assert test_results['exit_score'].shape[-1] == cfg['data']['seq_length']
    exit_scores_eval = OrderedDict({'time_0': OrderedDict(), 'time_1': OrderedDict(),
                                    'time_2': OrderedDict(), 'time_3': OrderedDict()})

    for i in range(test_results['exit_score'].shape[-1]):
        metrics, roc_plots, threshold, pred_labels = compute_single_exit_score_performance(test_results['exit_score'][:, i], test_results['target'])
        m, _, t, p = compute_single_exit_score_performance(test_results['exit_score'][:, -1], test_results['target'])
        exit_scores_eval['time_{}'.format(i)]['metrics'] = metrics
        exit_scores_eval['time_{}'.format(i)]['roc_plots'] = roc_plots
        exit_scores_eval['time_{}'.format(i)]['threshold'] = threshold
        exit_scores_eval['time_{}'.format(i)]['pred_labels'] = pred_labels
        exit_scores_eval['time_{}'.format(i)]['pred_scores'] = test_results['exit_score'][:, i]
        exit_scores_eval['time_{}'.format(i)]['targets'] = test_results['target']
        print(i, m, t)
    return exit_scores_eval


def average_fold_metrics(resutl_evals):
    """

    :param resutl_evals: 'folds' --> 'time_steps' --> 'metrics'
    :return:
    """

    res_metrics = {'accuracy': [], 'auc': [], 'recall': [],
                   'precision': [], 'f1-score': [], 'specificity': []}

    for fold, values in resutl_evals.items():
        metrics = [eval['metrics'] for time, eval in values.items()]
        for metric in metrics:
            res_metrics['accuracy'].append(metric['accuracy'])
            res_metrics['auc'].append(metric['auc'])
            res_metrics['recall'].append(metric['recall'])
            res_metrics['specificity'].append(metric['specificity'])
            res_metrics['f1-score'].append(metric['f1-score'])
            res_metrics['precision'].append(metric['precision'])

    for k, v in res_metrics:
        res_metrics[k] = (np.mean(v), np.std(v))

    return res_metrics


def plot_group_bars(metrics, plot_figure=True, save_dir=None):
    """

    :param metrics: time_0: {fold_0, fold_1, fold_2, fold_3, fold_4}, time_1: {}
    :param save_dir:
    :return:
    """
    values = OrderedDict()

    for time, metric in metrics.items():
        values[time] = OrderedDict()
        values[time].update({'accuracy': [np.mean([m['accuracy'] for m in metric])*100,
                            np.std([m['accuracy'] for m in metric])*100]})
        values[time].update({'auc': [np.mean([m['auc'] for m in metric])*100,
                                     np.std([m['auc'] for m in metric])*100]})
        values[time].update({'recall': [np.mean([m['recall'] for m in metric])*100,
                                        np.std([m['recall'] for m in metric])*100]})
        values[time].update({'precision': [np.mean([m['precision'] for m in metric])*100,
                                           np.std([m['precision'] for m in metric])*100]})
        values[time].update({'f1-score': [np.mean([m['f1-score'] for m in metric])*100,
                                          np.std([m['f1-score'] for m in metric])*100]})
        values[time].update({'specificity': [np.mean([m['specificity'] for m in metric])*100,
                                             np.std([m['specificity'] for m in metric])*100]})

    if plot_figure:
        width = 0.50
        labels = ['time_0', 'time_1', 'time_2', 'time_3']
        x = np.arange(len(labels))
        fig, ax = plt.subplots()
        acc = [val['accuracy'][0] for time, val in values.items()]
        acc_std = [val['accuracy'][1] for time, val in values.items()]
        rects1 = ax.bar(x - width * 2 / 5, acc, width / 5, color='tan', edgecolor='black', label='accuracy', yerr=acc_std, capsize=2)

        auc = [val['auc'][0] for time, val in values.items()]
        auc_std = [val['auc'][1] for time, val in values.items()]
        rects2 = ax.bar(x - width / 5, auc, width / 5, color='teal', edgecolor='black', label='Auc', yerr=auc_std, capsize=2)

        precision = [val['precision'][0] for time, val in values.items()]
        precision_std = [val['precision'][1] for time, val in values.items()]
        rects3 = ax.bar(x, precision, width / 5, color='salmon', edgecolor='black', label='precision', yerr=precision_std, capsize=2)

        recall = [val['recall'][0] for time, val in values.items()]
        recall_std = [val['recall'][1] for time, val in values.items()]
        rects4 = ax.bar(x + width / 5, recall, width / 5, color='silver', edgecolor='black', label='recall', yerr=recall_std, capsize=2)

        specificity = [val['specificity'][0] for time, val in values.items()]
        specificity_std = [val['specificity'][1] for time, val in values.items()]
        rects5 = ax.bar(x + width * 2 / 5, specificity, width / 5, color='plum', edgecolor='black', label='Specificity', yerr=specificity_std, capsize=2)

        ax.legend((rects1, rects2, rects3, rects4, rects5), ('Accuracy', 'AUC', 'Precision', 'Sensitivity', 'Specificity'),
                  bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylim([0, 90])
        ax.tick_params(axis="y", labelsize=14)
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'res_fig.png'))
        plt.show(block=1)
    return values


def plot_roc_curves(roc_plots, show_figures=True, savedir=None):
    folds_tpr = []
    base_fpr = np.linspace(0, 1, 101)

    color = ['b', 'g', 'r', 'k', 'y']
    if show_figures:
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
        print(os.path.join(savedir, 'res_fig.png'))

        plt.show()


def collect_results(result_evals, show_roc=False, savedir=None):
    """
    :param result_evals:
    :param show_roc:
    :return:
    """
    assert len(result_evals) == 5
    assert np.unique([len(result_evals['fold_0']), len(result_evals['fold_1']),
                      len(result_evals['fold_2']), len(result_evals['fold_3']),
                      len(result_evals['fold_4'])]) == len(result_evals['fold_0'])

    roc_plots = OrderedDict()
    metrics = OrderedDict()
    thresholds = OrderedDict()

    # collect results of each folds
    for t in range(len(result_evals['fold_0'])):
        roc_plots['time_{}'.format(t)] = []
        thresholds['time_{}'.format(t)] = []
        metrics['time_{}'.format(t)] = []
        for fold, value in result_evals.items():
            roc_plots['time_{}'.format(t)].append(value['time_{}'.format(t)]['roc_plots'])  # different times)
            thresholds['time_{}'.format(t)].append(value['time_{}'.format(t)]['threshold'])
            metrics['time_{}'.format(t)].append(value['time_{}'.format(t)]['metrics'])

    # group results of each time step
    values = plot_group_bars(metrics, plot_figure=True, save_dir=savedir)

    result_metrics = OrderedDict()
    result_metrics['roc_plots'] = roc_plots
    result_metrics['metrics'] = metrics
    result_metrics['thresholds'] = thresholds

    with open(os.path.join(savedir, 'time_metrics.json'), 'w') as f:
        json.dump(values, f, indent=4, sort_keys=True)
    return result_metrics


def cross_fold_validation(cfg, result_dir, best_model=True, data_length=1, show_roc=True):
    folds = glob(os.path.join(result_dir, 'fold*'))
    folds = sorted(folds, key=lambda x: x.split('/')[-1].split('_')[-1])

    model_file = cfg['model'] + '_best.model' if best_model else cfg['model'] + '_final.model'
    cfg['data']['seq_length'] = data_length
    # cfg['data']['seq_length'] = int(result_dir.split('/')[-1].split('_')[-1])
    print('current sequence length for evaluation is {}'.format(cfg['data']['seq_length']))

    # load data_split file
    run_exp_dir = cfg['run_exp']
    with open(os.path.join(run_exp_dir, 'data_setting', 'data_split.pkl'), 'rb') as f:
        data_split_file = pkl.load(f)

    # run five fold validation
    result_evals = OrderedDict()

    for i in range(len(folds)):
        start_time = time.time()
        cfg['fold'] = int(folds[i].split('_')[-1])
        if not os.path.exists(os.path.join(folds[i], 'exit_scores_evals.pt')):
            # load model
            model = get_model(cfg['model'], cfg['data']['channel'], 1, cfg['data']['seq_length']).to(device)
            saved_model = torch.load(os.path.join(folds[i], model_file), map_location=device)

            try:
                model.load_state_dict(saved_model['model_state_dict'])  # maybe need to change
            except RuntimeError:
                model_state_dict = convert_state_dict(saved_model['model_state_dict'])
                model.load_state_dict(model_state_dict)

            # evaluation
            print(os.path.join(folds[i], model_file))
            exit_scores_eval = evaluation(cfg, model, data_split_file)
            torch.save(exit_scores_eval, os.path.join(folds[i], 'exit_scores_eval.pt'))
            print('{}-th fold evaluated, using time {:.3f}'.format(i, time.time() - start_time))
            result_evals['fold_{}'.format(i)] = exit_scores_eval
        else:
            exit_scores_eval = torch.load(os.path.join(folds[i], 'exit_scores_eval.pt'))
            result_evals['fold_{}'.format(i)] = exit_scores_eval

    # early diagnosis module
    result_metrics = collect_results(result_evals, show_roc=False, savedir=result_dir)


if __name__ == "__main__":
    with open('/home/zyi/MedicalAI/Skin_lesion_prognosis/configs/skin_config.yml', 'r') as f:
        cfg = yaml.load(f)

    training_result_dirs = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/cnn_pool_s'
    cfg['model'] = 'cnn-pool-s'
    seq_length = [4]
    for length in seq_length:
        result_dir = os.path.join(training_result_dirs, 'seq_length_1')
        cross_fold_validation(cfg, result_dir, best_model=True, data_length=length, show_roc=True)
