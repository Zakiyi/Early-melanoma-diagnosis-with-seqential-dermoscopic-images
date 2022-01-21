import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.model_uitls import init_weights, convbnrelu


class Classifier(nn.Module):
    def __init__(self, in_channel, mid_channel, n_class):
        super(Classifier, self).__init__()
        # self.layer1 = nn.Sequential(nn.Linear(in_channel, mid_channel),
        #                             nn.BatchNorm1d(mid_channel),
        #                             nn.ReLU(inplace=False))

        self.layer2 = nn.Sequential(nn.BatchNorm1d(512),
                                    nn.Dropout(0.6),
                                    nn.Linear(in_channel, mid_channel),
                                    nn.BatchNorm1d(mid_channel),
                                    nn.ReLU(inplace=False))

        self.layer3 = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(mid_channel, n_class))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        assert len(x.shape) == 2

        # x = self.layer1(x.squeeze())
        x = self.layer2(x)
        pred = self.layer3(x)

        return pred


class ResNet_Kaggle(nn.Module):
    def __init__(self, in_channel=3, n_class=1, seq_length=1):
        super(ResNet_Kaggle, self).__init__()

        resnet = models.resnet34(True)

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = Classifier(resnet.fc.in_features, 32, n_class)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

        self.clss = nn.Sigmoid()

    def forward(self, x_seq):
        x = self.encoder(x_seq)  # batch x time_step x C
        x = self.classifier(x.squeeze(dim=-1).squeeze(dim=-1))

        return x


if __name__ == '__main__':

    input_seq = torch.rand(10, 3, 224, 224)

    resnet = ResNet_Pool()

    output = resnet(input_seq)
    print(output.shape)
