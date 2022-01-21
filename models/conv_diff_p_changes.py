import os
import pickle as pkl
import torch
import torch.nn as nn
import torchvision.models as models
from models.model_uitls import init_weights
import torchvision.transforms.functional as F
from models.model_uitls import GSOP_block
from data_proc.sequence_aug import Augmentations
from data_proc.ssd_datasplit import ssd_split
from data_proc.ssd_dataset import Skin_Dataset
from models.model_uitls import load_pretrained_resnet
from models.MPNCOV import MPNCOV
from dropblock import DropBlock2D, LinearScheduler


class Classifier(nn.Module):
    def __init__(self, in_channel, mid_channel, n_class):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(nn.ReLU(inplace=False),
                                    nn.Dropout(0.5),
                                    nn.Linear(in_channel, mid_channel),
                                    nn.BatchNorm1d(mid_channel),
                                    nn.ReLU(inplace=False))

        self.layer2 = nn.Sequential(nn.Dropout(0.6),
                                    nn.Linear(mid_channel, mid_channel),
                                    nn.BatchNorm1d(mid_channel),
                                    nn.ReLU(inplace=False))

        self.layer3 = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(mid_channel, n_class))

    def forward(self, x):
        x = x.squeeze()
        assert len(x.shape) == 2

        x = self.layer1(x.squeeze())
        x = self.layer2(x)
        pred = self.layer3(x)

        return pred


class TDNP(nn.Module):
    def __init__(self, in_channel=3, n_class=1, seq_length=1):
        super(TDNP, self).__init__()

        self.img_sbn_encoder = load_pretrained_resnet('resnet34')
        self.img_sbn_classifier = Classifier(self.img_sbn_encoder.fc_channel, 32, n_class)
        # self.img_sbn_classifier = nn.Sequential(nn.Dropout(0.5),
        #                                         nn.Linear(528, n_class))
        # self.gsop_block_0 = GSOP_block(64)    # conv_1
        # self.gsop_block_1 = GSOP_block(64)    # layer_1
        # self.gsop_block_2 = GSOP_block(128)   # layer_2
        # self.gsop_block_3 = GSOP_block(256)   # layer_3
        # self.drop_block = LinearScheduler(DropBlock2D(block_size=5, drop_prob=1.),
        #                                   start_value=1., stop_value=0.5, nr_steps=10)
        # self.gsop_block_4 = GSOP_block(512)   # layer_3
        # self.ISQRT_block = nn.Sequential(nn.Dropout(0.5),
        #                                  nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0),
        #                                  nn.BatchNorm2d(32),
        #                                  nn.ReLU(inplace=False),
        #                                  nn.Dropout(0.5))
        # self.spa_block_0 = SPT(64)
        # self.spa_block_1 = SPT(64)
        # self.spa_block_2 = SPT(128)
        # self.spa_block_3 = SPT(256)
        # self.spa_block_4 = SPT(512)
        # self.img_sbn_classifier = nn.Sequential(nn.Dropout(0.5),
        #                                         nn.Linear(self.img_sbn_encoder.fc_channel, n_class))

        # self.siamese_classifier = nn.Sequential(nn.Linear(2*5-1, 1),   # TODO: add dym_dfn_encoder feature
        #                                         nn.Sigmoid())
        #
        # nn.init.ones_(self.siamese_classifier[0].weight)
        # nn.init.zeros_(self.siamese_classifier[0].bias)

        for para in self.img_sbn_encoder.parameters():
            para.requires_grad = False
        #
        # for k, v in self.img_sbn_encoder.named_parameters():
        #     if 'se_block' in k:
        #         v.requires_grad = True
        #     else:
        #         v.requires_grad = False

        self.dym_dfn_encoder = load_pretrained_resnet('resnet34')
        self.dym_dfn_classifier = Classifier(self.dym_dfn_encoder.fc_channel, 32, n_class)

        # self.dym_dfn_classifier = nn.Sequential(nn.Dropout(0.5),
        #                                         nn.Linear(self.img_sbn_encoder.fc_channel, n_class))
        # for para in self.dym_dfn_encoder.parameters():
        #     para.requires_grad = False

    def img_subnetwork(self, img, f_prev=None):
        features, f_cur = self.img_sbn_encoder(img)

        # f_cur[0] = self.gsop_block_0(f_cur[0])
        # f_cur[1] = self.gsop_block_1(f_cur[1])
        # f_cur[2] = self.gsop_block_2(f_cur[2])
        # f_cur[3] = self.gsop_block_3(f_cur[3])

        # features = self.gsop_block_4(f_cur[4])
        # features = self.ISQRT_block(f_cur[-1])  # 3, 32, 10, 10
        # print(features.shape)
        # features = MPNCOV.CovpoolLayer(features)  # 3, 32, 32
        # features = MPNCOV.SqrtmLayer(features, 3)
        # features = MPNCOV.TriuvecLayer(features)
        # features = features.view(features.size(0), -1)

        # features = torch.cat([self.bn3(torch.nn.functional.adaptive_avg_pool2d(f_cur[3], (1, 1)).squeeze()),
        #                       self.bn4(torch.nn.functional.adaptive_avg_pool2d(f_cur[4], (1, 1)).squeeze())],
        #                      dim=1)
        # print(features.shape)
        pred_score = self.img_sbn_classifier(features.squeeze())

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
        for i in range(img_diff.shape[0]):
            img_diff[i, ...] = F.normalize(img_diff[i, ...], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        feature = self.dym_dfn_encoder(img_diff, feature_diff)

        if isinstance(feature, list):
            main_out = torch.flatten(feature[0], 1)
            pred_main = self.dym_dfn_classifier(main_out.squeeze())

            aux_out = torch.flatten(feature[-1], 1)
            pred_aux = self.aux_classifier(aux_out.squeeze())
            return [pred_main, pred_aux]
        else:
            feature = torch.flatten(feature, 1)
            pred = self.dym_dfn_classifier(feature.squeeze())

            return pred

    def forward(self, x_seq, p_index=False):
        Batch, Time_step, C, H, W = x_seq.shape

        if Time_step > 1:
            img_scores = []
            dif_scores = []

            img_score_0, features = self.img_subnetwork(x_seq[:, 0, ...])
            img_scores.append(img_score_0)   # img_score.shape = N * 1

            for i in range(Time_step-1):
                pred_scores, features, tmp_diff = self.img_subnetwork(x_seq[:, i+1, ...], features)

                score_diff = self.diff_network(x_seq[:, i+1, ...] - x_seq[:, i, ...], tmp_diff)
                img_scores.append(pred_scores)
                dif_scores.append(score_diff)

            img_sbn_scores = torch.cat(img_scores, dim=1)   # N * seq_length
            # img_sbn_scores = p_index * img_sbn_scores
            dym_dfn_scores = torch.cat(dif_scores, dim=1)

            # distance = 0
            # for j in range(len(img_embedding)-1):
            #     distance += nn.functional.pairwise_distance(img_embedding[j], img_embedding[j+1], p=1)

            '1. evenly average img_score and diff_score'
            # final_pred = torch.mean(torch.cat([img_sbn_scores, dym_dfn_scores], dim=1), dim=1)

            '2. learn weight the pred value of two modality data'
            # final_pred = self.siamese_classifier(torch.cat([img_sbn_scores, dym_dfn_scores], dim=1))

            '3. separately average img_score and diff_score'
            # img_sbn_scores = torch.mean(img_sbn_scores, dim=1)
            # dym_dfn_scores = torch.mean(dym_dfn_scores, dim=1)
            # final_pred =  0.8 * img_sbn_scores +  0.2*dym_dfn_scores

            '4. using p_index for diff_score to average'
            # img_sbn_scores = p_index * img_sbn_scores
            # img_sbn_scores = img_sbn_scores.sum(dim=1) / p_index.sum(dim=1)

            img_sbn_scores = torch.mean(img_sbn_scores, dim=1)   # N
            dym_dfn_scores = torch.mean(dym_dfn_scores, dim=1)

            img_sbn_scores = torch.sigmoid(img_sbn_scores)
            dym_dfn_scores = torch.sigmoid(dym_dfn_scores)
            # dym_dfn_scores = p_index[:, :-1] * dym_dfn_scores   # N * seq_length-1
            # dym_dfn_scores = dym_dfn_scores.sum(dim=1) / (p_index[:, :-1].sum(dim=1) + 1.0e-6)  # N

            final_pred = torch.stack([img_sbn_scores, dym_dfn_scores], dim=1).mean(dim=1)

        else:
            raise ValueError

        return final_pred


if __name__ == '__main__':
    input_seq = torch.rand(3, 4, 3, 320, 320)
    p_index = torch.ones(3, 4)
    resnet = TDNP()
    # for para in resnet.img_sbn_encoder.parameters():
    #     print(para.requires_grad)
    # resnet.encoder.module
    scores = resnet(input_seq, p_index)
    # print(scores.shape)
    # data_root = '/home/zyi/MedicalAI/Serial_Skin_data'
    # with open(os.path.join('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp',
    #                        'data_setting', 'data_split.pkl'), 'rb') as f:
    #     data_split_file = pkl.load(f)
    #
    # train_aug_parameters = {'affine': None,
    #                         'color_trans': {'brightness': (0.8, 1.2),
    #                                         'contrast': (0.8, 1.2),
    #                                         'saturation': (0.8, 1.2),
    #                                         'hue': (-0.1, 0.1)},
    #                         'normalization': {'mean': (0.485, 0.456, 0.406),
    #                                           'std': (0.229, 0.224, 0.225)},
    #                         'size': 320,
    #                         'scale': (0.8, 1.2),
    #                         'ratio': (0.9, 1.1)
    #                         }
    # augmentor = Augmentations(train_aug_parameters)
    # dataset = Skin_Dataset(data_root, data_split_file[0]['train'], seq_length=3,
    #                        transform=augmentor.transform, data_modality='MIC',
    #                        is_train=True)
    #
    # inputs = dataset[0]['image']['MIC'].unsqueeze(0)
    #
    # outputs = resnet(inputs)