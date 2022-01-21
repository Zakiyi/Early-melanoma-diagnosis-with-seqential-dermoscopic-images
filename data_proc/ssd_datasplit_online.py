import os
import numpy as np
import pickle as pkl
from glob import glob
from functools import reduce
from sklearn.model_selection import KFold
from collections import OrderedDict, Counter


class ssd_split():
    def __init__(self, case_list):
        self.case_list = case_list

    def keys_to_dir(self, case_dir_list, keys_list, repeat_items, is_train=False):
        assert len(case_dir_list) > 1
        # add repeated items
        for item in repeat_items:
            if np.any(keys_list == item):
                keys_list = np.delete(keys_list, np.where(keys_list == item))
                keys_list = np.append(keys_list, item + '(0)')
                if is_train:
                    keys_list = np.append(keys_list, item + '(1)')

        case_list = [os.path.join(path.split('/')[-2], key) for key in keys_list
                     for path in case_dir_list if key == os.path.basename(path)]

        return case_list

    def data_split(self):
        case_id = [os.path.basename(x).split('(')[0] for x in self.case_list]
        repeated_items = [item for item, count in Counter(case_id).items() if count > 1]
        # split the data for training and validation
        case_id = np.unique(case_id)
        case_id = np.random.permutation(case_id)
        kfold = KFold(n_splits=5, shuffle=True)
        splits = []

        for i, (train_idx, val_idx) in enumerate(kfold.split(case_id)):
            train_keys = np.array(case_id)[train_idx]
            val_keys = np.array(case_id)[val_idx]

            train_list = self.keys_to_dir(self.case_list, train_keys, repeated_items, True)
            val_list = self.keys_to_dir(self.case_list, val_keys, repeated_items, False)

            splits.append(OrderedDict())
            splits[-1]['train'] = train_list
            splits[-1]['val'] = val_list

            # print(len(splits[i]['train']), len(splits[i]['val']))
            # print(splits[i]['val'])

        return splits


if __name__ == "__main__":
    root = "/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned"
    exp_dir = '/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp'

    with open('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/data_setting/data_split.pkl', 'rb') as f:
        split = pkl.load(f)

    splits = ssd_split(case_list=split[0]['train']).data_split()
    print(len(splits[4]['val']))