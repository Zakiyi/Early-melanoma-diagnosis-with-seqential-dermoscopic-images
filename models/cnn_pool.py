import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.model_uitls import init_weights, TimeDistributed


class ResNet_Pool(nn.Module):
    def __init__(self, in_channel=3, n_classes=1, seq_length=1, pool_type='avg'):
        super(ResNet_Pool, self).__init__()

        if pool_type == 'avg':
            self.pool_fn = F.avg_pool1d
        elif pool_type == 'max':
            self.pool_fn = F.max_pool1d
        else:
            raise ValueError

        resnet = models.resnet34(True)

        self.encoder = TimeDistributed(nn.Sequential(*list(resnet.children())[:-1]))

        self.combiner = nn.Sequential(nn.Dropout(0.5),
                                      nn.Linear(resnet.fc.in_features, 64),
                                      nn.BatchNorm1d(64, momentum=0.01),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.6),
                                      nn.Linear(64, 64),
                                      nn.BatchNorm1d(64, momentum=0.01),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.5),
                                      nn.Linear(64, n_classes))

        for m in self.combiner.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

        self.clss = nn.Sigmoid()

    def forward(self, x_seq):
        x = self.encoder(x_seq)  # batch x time_step x C
        x = self.pool_fn(x.transpose(2, 1), kernel_size=x.shape[1], stride=1)  # batch x C x 1
        x = self.combiner(x.view(x.size(0), -1))

        return self.clss(x)


if __name__ == '__main__':
    # m = torch.load('/home/zyu/Desktop/Skin_lesion_prognosis v2/run_exp/256dropout(m10)0.2/resnet-pool_scheduled.model')
    # d = m['model_state_dict']
    input_seq = torch.rand(2, 10, 3, 224, 224)
    # print(type(input_seq))
    resnet = ResNet_Pool()
    n_parameters = sum(p.numel() for p in resnet.parameters())
    print(n_parameters)
    for k, v in resnet.named_parameters():
        print(k)
    # resnet.load_state_dict(d)
    # for para in resnet.encoder.parameters():
    #     para.requires_grad = False
    #
    # resnet.encoder.module
    # output = resnet(input_seq)
    # print(output.shape)
    #
    # print(resnet.combiner[2:])
    # for k, v in resnet.encoder.module[4].named_modules():
    #     print(k)
    # # for name, value in list(resnet.named_parameters()):
    # #     print(name)