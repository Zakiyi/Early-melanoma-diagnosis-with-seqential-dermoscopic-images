import torch
import numpy as np


class ClassErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class get_metrics(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.acc_cal_fun = None
        self.macc_cal_fun = None
        self.confusion_matrix = None

    def update(self, metrics_batch, metrics_overall):
        for name, value in metrics_batch.items():
            metrics_overall[name].append(metrics_batch[name])
        return metrics_overall
