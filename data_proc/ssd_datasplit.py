import os
import numpy as np
import pickle as pkl
from glob import glob
from functools import reduce
from sklearn.model_selection import KFold
from collections import OrderedDict, Counter


class ssd_split():
    def __init__(self, root, exp_dir):
        self.root = root
        self.exp_dir = exp_dir

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

    def data_split(self, save_file=False):
        # data collecting and checking
        case_list = glob(os.path.join(self.root, "SMI*", "*"))
        case_list = [x for x in case_list if os.path.isdir(x)]
        file_ext = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.bmp', '*.BMP']

        case_img_num = []       # store the num of images in sub-folders
        case_id = []
        for x in case_list:
            if os.path.isdir(x):
                # checking the image numbers of each sub-folder
                img_files = reduce(lambda files_x, files_y: files_x + files_y,
                                   list(map(lambda ext: glob(os.path.join(x, ext)), file_ext)))

                if len(img_files) < 1:
                    print(x)
                    raise ValueError("empty file folder should be removed!!!")
                else:
                    case_img_num.append(len(img_files))

                case_id.append(os.path.basename(x).split('(')[0])  # get the ID number of each case

        repeated_items = [item for item, count in Counter(case_id).items() if count > 1]

        assert len(case_list) == len(case_img_num) and len(case_list) == len(case_id)
        print('Total {} cases included'.format(len(case_id)))

        data_info = OrderedDict()
        data_info['case_path'] = case_list
        data_info['image_num'] = case_img_num
        data_info['case_id'] = case_id

        data_setting_dir = os.path.join(self.exp_dir, 'data_setting')
        if not os.path.exists(data_setting_dir):
            os.makedirs(data_setting_dir)

        with open(os.path.join(data_setting_dir, 'data_info.pkl'), 'wb') as f:
            pkl.dump(data_info, f)

        # split the data for training and validation
        case_id = np.unique(case_id)
        case_id = np.random.permutation(case_id)
        kfold = KFold(n_splits=5, shuffle=True)
        splits = []
        test_list = OrderedDict()

        test_id = case_id[np.linspace(0, len(case_id), len(case_id) // 5, endpoint=False, dtype=np.int)]
        test_list['test'] = self.keys_to_dir(case_list, test_id, repeated_items, False)
        # train_id = np.delete(case_id, np.linspace(0, len(case_id), len(case_id) // 5, endpoint=False, dtype=np.int))
        train_id = np.setdiff1d(case_id, test_id)
        splits.append(test_list)

        for i, (train_idx, val_idx) in enumerate(kfold.split(train_id)):
            train_keys = np.array(train_id)[train_idx]
            val_keys = np.array(train_id)[val_idx]
            print('train_val ', len(train_keys), len(val_keys))
            train_list = self.keys_to_dir(case_list, train_keys, repeated_items, True)
            val_list = self.keys_to_dir(case_list, val_keys, repeated_items, False)

            splits.append(OrderedDict())
            splits[-1]['train'] = train_list
            splits[-1]['val'] = val_list
        if save_file:
            with open(os.path.join(data_setting_dir, 'data_split.pkl'), 'wb') as f:
                pkl.dump(splits, f)

        return splits


if __name__ == "__main__":
    root = "/home/zyi/MedicalAI/HR_Serial_Skin_data_aligned"
    exp_dir = '/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp'

    ssd_split(root, exp_dir).data_split()

    with open('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/data_setting/data_split.pkl', 'rb') as f:
        split = pkl.load(f)

    # for fold in [1,2,3,4,5]:
    #     train = split[fold]['train']
    #     val = split[fold]['val']
    #
    #     for file in val:
    #         for f in split[0]['test']:
    #             if file == f:
    #                 print(fold)
    #                 print(f)
