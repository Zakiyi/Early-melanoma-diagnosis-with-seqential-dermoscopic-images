import os
import math
import pickle as pkl
import torch
import numpy as np
from torch.nn import init
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
from models.model_uitls import init_weights
import torch.nn.functional as F
from models.model_uitls import GSOP_block
from data_proc.sequence_aug_diff import Augmentations_diff
from data_proc.ssd_datasplit import ssd_split
from data_proc.ssd_dataset import Skin_Dataset
from models.model_uitls import load_pretrained_resnet, load_finetuned_resnet

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # remove `module`
        new_state_dict[name] = v
    return new_state_dict


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size=16, value_size=16):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, inputs):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i > j else 0 for i in range(inputs.shape[1])] for j in range(inputs.shape[1])])
        mask = torch.ByteTensor(mask).to(inputs.device)

        keys = self.linear_keys(inputs)        # shape: (N, T, key_size)
        query = self.linear_query(inputs)      # shape: (N, T, key_size)
        values = self.linear_values(inputs)    # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))   # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size, dim=1)   # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values)    # shape: (N, T, value_size)
        return torch.cat((inputs, temp), dim=2)    # shape: (N, T, in_channels + value_size)


class Classifier2(nn.Module):
    def __init__(self, in_channel, mid_channel, n_class=1, dropout=0.8):
        super(Classifier2, self).__init__()
        # solution 1
        self.layer1 = nn.Sequential(nn.Dropout2d(dropout),
                                    nn.AdaptiveAvgPool2d((1, 1)))

        self.layer2 = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(in_channel, mid_channel),
                                    nn.BatchNorm1d(mid_channel),
                                    nn.ReLU())

        self.layer3 = nn.Linear(mid_channel, 1)

        self.clss = n_class

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.layer1(x).squeeze()
        x = self.layer2(x)
        pred = self.layer3(x)
        return pred, x


def load_classifier_dict(classifier, pre_trained_model):
    model = torch.load(pre_trained_model, map_location='cpu')['model_state_dict']
    parameters_dict = OrderedDict()

    for k, v in model.items():
        if classifier in k:
            parameters_dict[k[19:]] = v

    return parameters_dict


class TDNHCSAN(nn.Module):
    def __init__(self, in_channel=3, n_class=1, seq_length=4, pre_trained_clss=None):
        super(TDNHCSAN, self).__init__()

        self.seq_length = seq_length
        assert seq_length > 2
        self.img_sbn_encoder = load_pretrained_resnet('resnet34')
        self.img_sbn_classifier = Classifier2(512, 16, dropout=0.5)
        self.dym_dfn_encoder = load_pretrained_resnet('resnet34')
        self.dym_dfn_classifier = Classifier2(512, 16, dropout=0.8)

        self.pre_train_clss = pre_trained_clss

        if self.pre_train_clss is not None:
            print('we are loading ', pre_trained_clss, ' as our base model!!!')
            img_sbn_classifier_dict = load_classifier_dict('img_sbn_classifier', pre_trained_clss)
            dym_dfn_classifier_dict = load_classifier_dict('dym_dfn_classifier', pre_trained_clss)

            self.img_sbn_classifier.load_state_dict(img_sbn_classifier_dict)
            self.dym_dfn_classifier.load_state_dict(dym_dfn_classifier_dict)

        self.attention_block = AttentionBlock(in_channels=32, key_size=16, value_size=16)
        self.clss_layer = nn.ModuleList([nn.Linear(48, 1),
                                         nn.Linear(48, 1),
                                         nn.Linear(48, 1),
                                         nn.Linear(48, 1),
                                         ])

        init_weights(self.clss_layer, init_type='kaiming')
        init_weights(self.attention_block, init_type='kaiming')

        for para in self.dym_dfn_encoder.parameters():
            para.requires_grad = False

        for para in self.img_sbn_encoder.parameters():
            para.requires_grad = False

    def img_subnetwork(self, img, f_prev=None):
        features, f_cur = self.img_sbn_encoder(img)
        pred_score, x = self.img_sbn_classifier(features.squeeze())

        if f_prev is None:
            return pred_score, f_cur, x
        else:
            feature_diff = []
            assert len(f_cur) == len(f_prev)
            for l in range(len(f_cur)):
                feature_diff.append(f_cur[l] - f_prev[l])   # TODO: add conv to obtain the feature diff
            return pred_score, f_cur, feature_diff, x

    def diff_network(self, img_diff, feature_diff):
        # img_diff shape = N, C, H, W
        feature = self.dym_dfn_encoder(img_diff, feature_diff)
        pred, x = self.dym_dfn_classifier(feature.squeeze())

        return pred, x

    def forward(self, img_seq, diff_seq, p_index=None):

        self.dym_dfn_encoder.eval()
        self.img_sbn_encoder.eval()

        # if self.pre_train_clss is not None:
        self.dym_dfn_classifier.eval()
        self.img_sbn_classifier.eval()

        try:
            Batch, Time_step, C, H, W = img_seq.shape
        except ValueError:
            img_seq = img_seq.unsqueeze(0)
            diff_seq = diff_seq.unsqueeze(0)
            Batch, Time_step, C, H, W = img_seq.shape

        if Time_step > 1:
            img_score_0, features, x_0 = self.img_subnetwork(img_seq[:, 0, ...])
            exit_scores = []
            spatial_scores = []
            temporal_scores = []

            exit_scores.append(img_score_0)
            spatial_scores.append(img_score_0)

            exit_features = []
            exit_feature_scores = []

            exit_features.append(torch.cat([x_0, torch.zeros_like(x_0)], dim=1))

            for i in range(Time_step-1):
                score_img, features, tmp_diff, x_i = self.img_subnetwork(img_seq[:, i+1, ...], features)
                score_diff, x_d = self.diff_network(diff_seq[:, i, ...], tmp_diff)

                temporal_scores.append(score_diff)  # p_index
                spatial_scores.append(score_img)

                exit_features.append(torch.cat([x_i, x_d], dim=1))

            exit_features = torch.stack(exit_features, dim=0).transpose(0, 1)  # TNC --> NTC
            exit_features = self.attention_block(exit_features)  # NTC

            for j in range(exit_features.shape[1]):
                exit_feature_scores.append(self.clss_layer[j](exit_features[:, j, :]))

            img_sbn_scores = torch.cat(spatial_scores, dim=1)  # N * seq_length
            img_sbn_scores = torch.mean(img_sbn_scores, dim=1)  # N

            dym_dfn_scores = torch.cat(temporal_scores, dim=1)
            dym_dfn_scores = torch.mean(dym_dfn_scores, dim=1)

            average_scores = torch.stack([img_sbn_scores, dym_dfn_scores], dim=1).mean(dim=1)

            final_pred = [torch.sigmoid(img_sbn_scores).squeeze(),
                          torch.sigmoid(dym_dfn_scores).squeeze(),
                          torch.sigmoid(average_scores).squeeze()]

        else:
            raise ValueError

        return final_pred, spatial_scores, exit_features, exit_feature_scores


if __name__ == '__main__':
    input_seq = torch.rand(3, 4, 3, 320, 320)
    p_index = torch.ones(3, 4)
    resnet = TDNHCSA(seq_length=4)

    # state_dict = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/cnn-diff-hc_MIC/conv_last_layer/seq_length_3/'
    #                         'fold_0/cnn-diff-hc_best.model')['model_state_dict']
    # resnet.load_state_dict(convert_state_dict(state_dict))

    data_root = '/home/zyi/MedicalAI/HR_Serial_Skin_data'
    with open(os.path.join('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp',
                           'data_setting', 'data_split.pkl'), 'rb') as f:
        data_split_file = pkl.load(f)

    train_aug_parameters = {'affine': None,
                            'flip': True,
                            'color_trans': {'brightness': (1.0, 1.0),
                                            'contrast': (1.0, 1.0),
                                            'saturation': (1.0, 1.0),
                                            'hue': (-0.0001, 0.0001)},
                            'normalization': {'mean': (0.485, 0.456, 0.406),
                                              'std': (0.229, 0.224, 0.225)},
                            'size': 320,
                            'scale': (1.0, 1.0),
                            'ratio': (1.0, 1.0)
                            }
    augmentor = Augmentations_diff(train_aug_parameters, color_recalibration=True, test_mode=False)
    dataset = Skin_Dataset(data_root, data_split_file[0]['train'], seq_length=4, transform=augmentor.transform,
                           data_modality='MIC', is_train=True)

    inputs = dataset[20]['image']['MIC']
    image_seq_1 = inputs['images']
    diff_image_seq_1 = inputs['diff_images']

    inputs = dataset[26]['image']['MIC']
    image_seq_2 = inputs['images']
    diff_image_seq_2 = inputs['diff_images']

    p_index = torch.stack((dataset[20]['image']['p_index'], dataset[26]['image']['p_index']), dim=0)

    image_seq = torch.stack((image_seq_1, image_seq_2), dim=0)
    diff_image_seq = torch.stack((diff_image_seq_1, diff_image_seq_2), dim=0)

    outputs, spatial_scores, temporal_scores, _ = resnet(image_seq, diff_image_seq, p_index=p_index)



