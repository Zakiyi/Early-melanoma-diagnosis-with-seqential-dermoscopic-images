import os
import pickle as pkl
import torch
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
# from models.ResNet_diff import load_pretrained_resnet_diff
# from misc.loss_function import Ranking_Loss, Self_Distillation, BCE_Loss


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


class Classifier(nn.Module):
    def __init__(self, in_channel, mid_channel, n_class=1, dropout=0.5):
        super(Classifier, self).__init__()
        self.layer1 = nn.Dropout2d(dropout)
        self.layer2 = nn.AdaptiveAvgPool2d((1, 1))
        self.layer3 = nn.Linear(in_channel, mid_channel)
        self.layer4 = nn.Linear(mid_channel, 1)

        self.clss = n_class

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):

        x = self.layer2(self.layer1(x)).squeeze()
        x = self.layer3(x)
        pred = self.layer4(F.relu(x))
        return pred


class Classifier2(nn.Module):
    def __init__(self, in_channel, mid_channel, n_class=1, dropout=0.8):
        super(Classifier2, self).__init__()
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
        return pred


class TDNHC0(nn.Module):
    def __init__(self, in_channel=3, n_class=1, seq_length=4):
        super(TDNHC0, self).__init__()

        self.seq_length = seq_length
        assert seq_length > 2
        self.img_sbn_encoder = load_pretrained_resnet('resnet34')
        self.img_sbn_classifier = Classifier(512, 32, dropout=0.5)
        self.dym_dfn_encoder = load_pretrained_resnet('resnet34')
        self.dym_dfn_classifier = Classifier(512, 32, dropout=0.8)

        # self.exit_layer = nn.Linear(32, 1)
        # init_weights(self.exit_layer, init_type='kaiming')

        for para in self.dym_dfn_encoder.parameters():
            para.requires_grad = False

        for para in self.img_sbn_encoder.parameters():
            para.requires_grad = False

        # self.avg_weights = torch.nn.Parameter(torch.tensor([0.8, 0.2]), requires_grad=True)

    def img_subnetwork(self, img, f_prev=None):
        features, f_cur = self.img_sbn_encoder(img)
        pred_score = self.img_sbn_classifier(features)

        if f_prev is None:
            return pred_score, f_cur
        else:
            feature_diff = []
            assert len(f_cur) == len(f_prev)
            for l in range(len(f_cur)):
                feature_diff.append(f_cur[l] - f_prev[l])   # TODO: add conv to obtain the feature diff
            return pred_score, f_cur, feature_diff

    def diff_network(self, img_diff, feature_diff):
        # img_diff shape = N, C, H, W
        feature = self.dym_dfn_encoder(img_diff, feature_diff)
        pred = self.dym_dfn_classifier(feature)

        return pred

    def forward(self, img_seq, diff_seq, p_index=None):

        self.dym_dfn_encoder.eval()
        self.img_sbn_encoder.eval()

        try:
            Batch, Time_step, C, H, W = img_seq.shape
        except ValueError:
            img_seq = img_seq.unsqueeze(0)
            diff_seq = diff_seq.unsqueeze(0)
            Batch, Time_step, C, H, W = img_seq.shape

        if Time_step > 1:
            img_score_0, features = self.img_subnetwork(img_seq[:, 0, ...])

            spatial_scores = []
            spatial_scores.append(img_score_0)

            temporal_scores = []

            """directly mean"""
            exit_basemen_scores = []

            for i in range(Time_step-1):
                score_img, features, tmp_diff = self.img_subnetwork(img_seq[:, i+1, ...], features)
                score_diff = self.diff_network(diff_seq[:, i, ...], tmp_diff)

                temporal_scores.append(score_diff)  # p_index
                spatial_scores.append(score_img)

            img_sbn_scores = torch.cat(spatial_scores, dim=1)  # N * seq_length
            dym_dfn_scores = torch.cat(temporal_scores, dim=1)
            average_scores = torch.cat([img_sbn_scores, dym_dfn_scores], dim=1)

            # add normalize
            # average_scores = self.exit_layer(average_scores)

            img_sbn_scores = torch.mean(img_sbn_scores, dim=1) # N
            dym_dfn_scores = torch.mean(dym_dfn_scores, dim=1)

            average_scores = torch.stack([img_sbn_scores, dym_dfn_scores], dim=1).mean(dim=1)

            final_pred = [torch.sigmoid(img_sbn_scores).squeeze(),
                          torch.sigmoid(dym_dfn_scores).squeeze(),
                          torch.sigmoid(average_scores).squeeze()]

        else:
            raise ValueError
        return final_pred, spatial_scores, temporal_scores, exit_basemen_scores


if __name__ == '__main__':
    input_seq = torch.rand(3, 4, 3, 320, 320)
    p_index = torch.ones(3, 4)
    resnet = TDNHC0(seq_length=4)

    state_dict = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/run_exp/cnn-diff-hc_MIC/'
                            'orginal_random_seq/without_alignment_data/seq_length_4/fold_0/cnn-diff-hc_best.model')['model_state_dict']
    resnet.load_state_dict(convert_state_dict(state_dict))

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
    dataset = Skin_Dataset(data_root, data_split_file[1]['train'], seq_length=4, transform=augmentor.transform,
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
    # print(outputs)
    # print(torch.sigmoid(spatial_scores[0]),
    #       torch.sigmoid(spatial_scores[1]),
    #       torch.sigmoid(spatial_scores[2]))
    #
    # # loss = Ranking_Loss(spatial_scores, torch.tensor([0, 1]))
    # loss = Self_Distillation(spatial_scores, torch.tensor([0, 1]))
    # print(loss)

    # model = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/cnn-diff-hc_MIC/temporal_pretrained/wo_index_mean/seq_length_4/fold_0/cnn-diff-hc_best.model'
    #                    , map_location='cpu')['model_state_dict']
    #
    # resnet.load_state_dict(convert_state_dict(model))


