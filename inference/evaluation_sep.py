import os
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
from sklearn.metrics import roc_curve, confusion_matrix
from models import convert_state_dict

threshold = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['benign', 'malignant']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']


def get_model_output(model, data, modality='MIC', device='cuda', length=1):
    # print('inputs: ', inputs.shape)
    img_seq = data['image'][modality][:, :length, ...].unsqueeze(dim=1)
    print('image ', img_seq.shape)
    outputs = model(img_seq.to(device))  # p_index = N*seq_length

    return outputs


def evaluation(cfg, model, data_split_file, length):
    """

    :param cfg:
    :param model:
    :param data_split_file:
    :param test_aug:
    :return:
    """

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

    test_augmentor = Augmentations(test_aug_parameters, test_mode=True)
    # cfg['data']['seq_length'] always load 4 images
    assert cfg['data']['seq_length'] == 4
    dataset = Skin_Dataset(cfg['data']['root'], data_split_file[cfg['fold']]['val'], seq_length=cfg['data']['seq_length'],
                           transform=test_augmentor.transform, data_modality=cfg['data']['modality'], is_train=False,
                           test_mode=True)

    test_results = OrderedDict({'target': [], 'pred_score': [], 'pred_label': []})
    model.eval()

    with torch.no_grad():
        for data in iter(dataset):
            outputs = get_model_output(model, data, cfg['data']['modality'], length=length)

            pred_score = outputs.detach().cpu().numpy()  # (Ncrops * seq_length, 1)
            pred_score = np.mean(pred_score, axis=0)
            print('pred_score: ', pred_score)
            print('target: ', data['target'].detach().cpu().numpy())
            test_results['target'] += [data['target'].detach().cpu().numpy()]
            test_results['pred_score'] += [pred_score.squeeze()]

    fpr, tpr, thresholds = roc_curve(test_results['target'], test_results['pred_score'])
    """choose the best threshold"""
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    roc_t_threshold = list(roc_t['threshold'])[0]
    test_results['pred_label'] = list(map(lambda x: 1 if x > roc_t_threshold else 0, test_results['pred_score']))

    """compute metrics"""
    accuracy = metrics.accuracy_score(test_results['target'], test_results['pred_label'])
    auc = metrics.roc_auc_score(test_results['target'], test_results['pred_score'])
    recall = metrics.recall_score(test_results['target'], test_results['pred_label'])
    precision = metrics.precision_score(test_results['target'], test_results['pred_label'])
    f1_score = metrics.f1_score(test_results['target'], test_results['pred_label'])

    tn, fp, fn, tp = confusion_matrix(test_results['target'], test_results['pred_label']).ravel()
    specificity = tn / (tn + fp)

    return {'accuracy': accuracy, 'auc': auc, 'recall': recall, 'precision': precision,
            'f1-score': f1_score, 'specificity': specificity, 'threshold': roc_t_threshold}, (fpr, tpr), test_results


def cross_fold_validation(cfg, result_dir, best_model=True):
    folds = glob(os.path.join(result_dir, 'fold*'))
    folds = sorted(folds, key=lambda x: x.split('/')[-1].split('_')[-1])

    length = np.int(result_dir.split('_')[-1])
    model_file = cfg['model'] + '_best.model' if best_model else cfg['model'] + '_final.model'
    print('current sequence length for evaluation is {}'.format(cfg['data']['seq_length']))

    # load data_split file
    run_exp_dir = cfg['run_exp']
    with open(os.path.join(run_exp_dir, 'data_setting', 'data_split.pkl'), 'rb') as f:
        data_split_file = pkl.load(f)

    # run five fold validation
    result_metrics = OrderedDict()
    folds_tpr = []
    base_fpr = np.linspace(0, 1, 101)
    color = ['b', 'g', 'r', 'k', 'y']
    for i in range(len(folds)):
        start_time = time.time()
        cfg['fold'] = int(folds[i].split('_')[-1])

        # load model
        model = get_model(cfg['model'], cfg['data']['channel'], n_classes=1, seq_length=length).to(device)
        saved_model = torch.load(os.path.join(folds[i], model_file))

        try:
            model.load_state_dict(saved_model['model_state_dict'])  # maybe need to change
        except RuntimeError:
            model_state_dict = convert_state_dict(saved_model['model_state_dict'])
            model.load_state_dict(model_state_dict)

        # evaluation
        print(os.path.join(folds[i], model_file))

        result_metrics['fold_{}'.format(cfg['fold'])], (fpr, tpr), test_res = evaluation(cfg, model, data_split_file, length)
        torch.save(test_res, os.path.join(folds[i], 'test_res.pt'))
        plt.plot(fpr, tpr, color[i], alpha=0.1)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        folds_tpr.append(tpr)
        print('{}-th fold evaluated, using time {:.3f}'.format(i, time.time() - start_time))

    np.save(os.path.join(result_dir, 'roc_plot_' + model_file.split('_')[-1].split('.')[0]), folds_tpr)
    with open(os.path.join(result_dir, model_file.split('.')[0] + '_evaluation.json'), 'w') as f:
        json.dump(result_metrics, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    with open('/home/zyi/MedicalAI/Skin_lesion_prognosis/configs/skin_config.yml', 'r') as f:
        cfg = yaml.load(f)

    training_result_dir = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/CNN_features/single_img_cnn_ham'
    seq_length = [2]
    cfg['model'] = 'cnn-pool-s'
    cfg['data']['seq_length'] = 4
    for length in seq_length:
        result_dir = os.path.join(training_result_dir, 'seq_length_{}'.format(length))
        cross_fold_validation(cfg, result_dir, best_model=True)

