import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict
from models.model_uitls import TimeDistributed
from models.model_uitls import init_weights, load_pretrained_resnet


class ResNet_HM10000(nn.Module):
    def __init__(self):
        super(ResNet_HM10000, self).__init__()

        resnet = models.resnet34(True)
        self.layer0 = nn.Sequential(resnet.conv1,
                                    resnet.bn1,
                                    resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dropout, pre_trained_model=None):
        super(Encoder, self).__init__()

        if pre_trained_model is None:
            print('no pre_trained model')
            self.base = nn.Sequential(*list(models.resnet34(True).children())[:-2])
        else:
            model_state_dict = convert_state_dict(torch.load(pre_trained_model, map_location='cpu')['model_state_dict'])
            self.base = ResNet_HM10000()
            self.base.load_state_dict(model_state_dict)

        self.features = nn.Sequential(nn.Dropout2d(dropout),
                                        nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        x = self.base(x)
        x = self.features(x)
        return x


class Classifier(nn.Module):
    def __init__(self, in_channel, mid_channel, n_class, dropout=0.5):
        super(Classifier, self).__init__()

        self.layer1 = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(in_channel, mid_channel),
                                    nn.BatchNorm1d(mid_channel),
                                    nn.ReLU())

        self.layer2 = nn.Linear(mid_channel, n_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'layer' in k:
            name = k[7:]  # remove `module`
            new_state_dict[name] = v
    return new_state_dict


class ResNet_Pool_h(nn.Module):
    def __init__(self, in_channel=3, n_classes=1, seq_length=1, pre_trained=None):
        super(ResNet_Pool_h, self).__init__()

        if pre_trained is not None:
            self.encoder = Encoder(0.5, pre_trained)
        else:
            self.encoder = Encoder(dropout=0.5)

        for para in self.encoder.parameters():
            para.requires_grad = False

        self.classifier = Classifier(512, 32, n_classes)

        self.clss = nn.Sigmoid()

    def forward(self, x_seq):
        assert len(x_seq.size()) >= 4
        if len(x_seq.size()) == 4:
            x_seq = x_seq.unsqueeze(0)

        batch, time_step, C, H, W = x_seq.shape
        features = []
        for t in range(time_step):
            x = self.encoder(x_seq[:, t, ...])
            features.append(x)  # N*C

        features = torch.stack(features, dim=1)  # N*T*C
        x = features.mean(dim=1)
        x = self.classifier(x.squeeze())

        return self.clss(x)


if __name__ == '__main__':
    # m = torch.load('/home/zyu/Desktop/Skin_lesion_prognosis v2/run_exp/256dropout(m10)0.2/resnet-pool_scheduled.model')
    # d = m['model_state_dict']
    input_seq = torch.rand(10, 3, 3, 224, 224)
    # print(type(input_seq))
    resnet = ResNet_Pool_h(pre_trained='/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/HAM_pre_trained/cnn-hm10000_best.model')
    n_parameters = sum(p.numel() for p in resnet.parameters())
    print(n_parameters)
    # resnet.load_state_dict(d)
    # for para in resnet.encoder.parameters():
    #     para.requires_grad = False
    #
    # resnet.encoder.module
    # output = resnet(input_seq)
    #print(output.shape)

    # print(resnet.combiner[2:])
    # for k, v in resnet.encoder.module[4].named_modules():
    #     print(k)
    # # for name, value in list(resnet.named_parameters()):
    # #     print(name)