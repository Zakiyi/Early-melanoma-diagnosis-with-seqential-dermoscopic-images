import os
import numpy as np
import json
from glob import glob
import matplotlib.pyplot as plt
from collections import OrderedDict


class ResultProcessor:
    def __init__(self, res_dir, mode='best', is_final_test=False):
        self.res_dir = res_dir
        self.mode = mode
        self.avg_metrics = OrderedDict()

        if self.mode == 'best':
            self.file_name = '*best_evaluation.json'
        else:
            self.file_name = '*final_evaluation.json'

        result_file = glob(os.path.join(res_dir, self.file_name))[0]

        print('we are using the ' + result_file + ' for compute the average metrics!!!')
        with open(result_file, 'rb') as f:
            self.fold_metrics = json.load(f)

        self.res_metrics = {'accuracy': [], 'auc': [], 'recall': [], 'precision': [], 'f1-score': [], 'specificity':[]}

    def compute_mean_std(self, metrics):
        mean = np.mean(metrics)
        std = np.std(metrics)

        return (mean, std)

    def average_fold_metrics(self):
        for fold, metrics in self.fold_metrics.items():
            for key, value in metrics.items():
                try:
                    self.res_metrics[key].append(value)
                except KeyError:
                    pass

        print(self.res_metrics)
        self.avg_metrics['accuracy'] = self.compute_mean_std(self.res_metrics['accuracy'])
        self.avg_metrics['auc'] = self.compute_mean_std(self.res_metrics['auc'])
        self.avg_metrics['recall'] = self.compute_mean_std(self.res_metrics['recall'])
        self.avg_metrics['precision'] = self.compute_mean_std(self.res_metrics['precision'])
        self.avg_metrics['f1-score'] = self.compute_mean_std(self.res_metrics['f1-score'])
        self.avg_metrics['specificity'] = self.compute_mean_std(self.res_metrics['specificity'])
        print(self.avg_metrics)

        with open(os.path.join(fold_result_dir, 'avg_metrics_' + self.file_name), 'w') as f:
            json.dump(self.avg_metrics, f, indent=4, sort_keys=True)

        return self.res_metrics, self.avg_metrics


if __name__ == '__main__':
    train_result_dirs = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/run_exp/cnn-diff-hc_MIC/orginal_random_seq/without_alignment_aligned'
    seq_length = [4]

    # for train_result_dir in glob(os.path.join(train_result_dirs, '*')):
    #     for length in seq_length:
    #         fold_result_dir = os.path.join(train_result_dir, 'seq_length_{}'.format(length))
    #         res_process = ResultProcessor(fold_result_dir, mode='best')
    #         res_metrics, avg_metrics = res_process.average_fold_metrics()
    #         with open(os.path.join(fold_result_dir, 'avg_metrics_' + str(res_process.mode) + '.json'), 'w') as f:
    #             json.dump(avg_metrics, f, indent=4, sort_keys=True)

    for length in seq_length:
        fold_result_dir = os.path.join(train_result_dirs, 'seq_length_{}'.format(length))
        res_process = ResultProcessor(fold_result_dir, mode='best', is_final_test=False)
        res_metrics, avg_metrics = res_process.average_fold_metrics()
        # with open(os.path.join(fold_result_dir, 'avg_metrics_' + str(res_process.mode) + '.json'), 'w') as f:
        #     json.dump(avg_metrics, f, indent=4, sort_keys=True)

