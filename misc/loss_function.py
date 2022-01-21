import torch
import functools
import torch.nn as nn
import torch.nn.functional as F

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse


def BCE_Loss(input, target):
    if input.shape != target.shape:
        input = input.view(target.shape)

    if target.dtype != input.dtype:
        target = target.type(input.dtype)

    loss = F.binary_cross_entropy(input, target)

    return loss


def BCE_Index_Loss(input, target, p_index=None):
    print('we are using BCE index loss !!!')

    if input.shape != target.shape:
        input = input.view(target.shape)

    if target.dtype != input.dtype:
        target = target.type(input.dtype)

    weights = p_index[:, -1]
    # p_index[p_index > 0] = 1.0
    # p_index[p_index == 0] = 0.
    # weights = p_index
    loss = F.binary_cross_entropy(input, target, weight=weights, reduction='mean')

    # if p_index is None:
    #     loss = F.binary_cross_entropy(input, target)
    # else:
    #     p_index = p_index[:, :-1].sum(dim=1)
    #     p_index[p_index > 0] = 1
    #
    #     input = input[p_index == 1]
    #     target = target[p_index == 1]
    #     loss = F.binary_cross_entropy(input, target)

    return loss


def Self_Distillation(inputs, target=None, alpha=0.4, p_index=None):
    """
    loss function for Knowledge Distillation (KD)
    outputs: list[N, N, N]
    labels: N

    """
    loss = 0
    assert len(inputs) > 1
    weights = p_index[:, len(inputs):]
    for i in range(len(inputs)-1):
        l = F.kl_div(torch.log(torch.sigmoid(inputs[i])), torch.sigmoid(inputs[-1]).detach(), reduction='batchmean')
        b = (1. - alpha) * F.binary_cross_entropy(torch.sigmoid(inputs[i].squeeze()), target.float(),
                                                  weight=weights[:, i], reduction='mean') + alpha * l

        loss += b

    return loss / (len(inputs)-1)


def Ranking_Loss(input, target, margin=(0.05, -0.01), type='later', p_index=None):
    """
    :param input: list[N, N, N....]
    :param target: N
    :return:
    """
    assert torch.equal(torch.unique(target), torch.tensor([0, 1], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([0], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([1], dtype=torch.int64).to(target.device))

    print('we are using margin of {} and {}'.format(margin[0], margin[1]))

    loss = 0.
    mel = torch.cat(input, dim=1)[target == 1, :]   # Batch * seq_length
    ben = torch.cat(input, dim=1)[target == 0, :]

    mel_p = p_index[target == 1, :]
    ben_p = p_index[target == 0, :]
    # loss += F.binary_cross_entropy(torch.sigmoid(input[0].squeeze()), target.float(), reduction='mean')
    # loss += Margin_Index_Loss(input[0], target)  # fix
    print('mel ', torch.sigmoid(mel))
    # for melanoma case
    print('ben ', torch.sigmoid(ben))
    if mel.shape[0] > 1:
        min_mel, _ = torch.min(mel, dim=1)
        weight_mel = torch.max(torch.zeros_like(min_mel), 0.8 * torch.ones_like(min_mel) - torch.sigmoid(min_mel))
        weight_mel = torch.sign(weight_mel).detach()

        if type == 'base':
            # mel = F.softmax(mel, dim=1)
            # mel = torch.exp(mel) / torch.exp(mel).norm(p=1, dim=1, keepdim=True).detach()

            print('we are using ranking loss with softmax!!!')
            for i in range(0, mel.shape[-1] - 1):
                for j in range(i+1, mel.shape[-1]):
                    r = F.margin_ranking_loss(mel[:, i].squeeze().sigmoid().detach(),
                                              mel[:, j].squeeze().sigmoid(),
                                              -torch.ones_like(weight_mel), margin=margin[0])

                    loss += r

        elif type == 'with_softmax_j-i':
            print('we are using ranking loss without softmax!!!')
            # mel = F.softmax(mel, dim=1)
            # mel = torch.exp(mel) / torch.exp(mel).norm(p=1, dim=1, keepdim=True).detach()
            print('we are using ranking loss with softmax!!!')
            for i in range(0, mel.shape[-1] - 1):
                for j in range(i + 1, mel.shape[-1]):
                    r = F.margin_ranking_loss(mel[:, i].squeeze().sigmoid().detach(),
                                              mel[:, j].squeeze().sigmoid(),
                                              -torch.ones_like(weight_mel) * mel_p[:, i], margin=margin[0]*(j-i))

                    loss += r
        elif type == 'both':
            # mel = F.softmax(mel, dim=1)
            # mel = torch.exp(mel) / torch.exp(mel).norm(p=1, dim=1, keepdim=True).detach()
            print('we are using ranking loss with softmax!!!')
            for i in range(0, mel.shape[-1] - 1):
                for j in range(i + 1, mel.shape[-1]):
                    r = F.margin_ranking_loss(mel[:, i].squeeze().sigmoid(),
                                              mel[:, j].squeeze().sigmoid(),
                                              -torch.ones_like(weight_mel), margin=margin[0])

                    loss += r

        else:
            raise ValueError

    # for benign case
    if ben.shape[0] > 1:
        max_ben, _ = torch.max(ben, dim=1)
        weight_ben = torch.max(torch.zeros_like(max_ben), torch.sigmoid(max_ben) - 0.2 * torch.ones_like(max_ben))
        weight_ben = torch.sign(weight_ben).detach()

        if type == 'base':
            # ben = F.softmax(ben, dim=1)
            # ben = torch.exp(ben) / torch.exp(ben).norm(p=1, dim=1, keepdim=True).detach()
            print('we are using ranking loss with softmax!!!')
            for i in range(0, ben.shape[-1] - 1):
                for j in range(i+1, ben.shape[-1]):
                    r = F.margin_ranking_loss(ben[:, i].squeeze().sigmoid().detach(),
                                              ben[:, j].squeeze().sigmoid(),
                                              torch.ones_like(weight_ben), margin=margin[1])  # x2-x1, only update x2
                    loss += r

        elif type == 'with_softmax_j-i':
            print('we are using ranking loss with softmax!!!')
            # ben = F.softmax(ben, dim=1)
            # ben = torch.exp(ben) / torch.exp(ben).norm(p=1, dim=1, keepdim=True).detach()
            print('we are using ranking loss with softmax!!!')
            for i in range(0, ben.shape[-1] - 1):
                for j in range(i + 1, ben.shape[-1]):
                    r = F.margin_ranking_loss(ben[:, i].squeeze().sigmoid().detach(),
                                              ben[:, j].squeeze().sigmoid(),
                                              torch.ones_like(weight_ben)*ben_p[:, i], margin=margin[1]*(j-i))  # x2-x1, only update x2
                    loss += r
        elif type == 'both':
            # ben = F.softmax(ben, dim=1)
            # ben = torch.exp(ben) / torch.exp(ben).norm(p=1, dim=1, keepdim=True).detach()
            print('we are using ranking loss with softmax!!!')
            for i in range(0, ben.shape[-1] - 1):
                for j in range(i + 1, ben.shape[-1]):
                    r = F.margin_ranking_loss(ben[:, i].squeeze().sigmoid(),
                                              ben[:, j].squeeze().sigmoid(),
                                              torch.ones_like(weight_ben), margin=margin[1])  # x2-x1, only update x2
                    loss += r
        else:
            raise ValueError

    # print('within rank ', loss)
    margin_loss = Margin_Loss(input, target, p_index=p_index)
    # print('within margin ', margin_loss)
    loss = margin_loss + loss/(len(input)-1)
    return loss


def get_ext_value(input, target):
    """
    :param input: list[N, N, N....]
    :param target: N belongs to 0 or 1
    :return:
    """
    assert torch.equal(torch.unique(target), torch.tensor([-1, 1], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([-1], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([1], dtype=torch.int64).to(target.device))

    input = torch.cat(input, dim=1)  # Batch * Seq

    out = []
    for i in range(len(target)):
        if target[i] > 0:
            v = torch.min(input[i, :])
            out.append(torch.sigmoid(v))
        else:
            v = torch.max(input[i, :])
            out.append(torch.sigmoid(v))

    out = torch.stack(out)

    return out


def Margin_Loss(inputs, target, p_index=None):
    """
    :param input: list[N, N, N....]
    :param target: N
    :return:
    """
    assert torch.equal(torch.unique(target), torch.tensor([0, 1], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([0], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([1], dtype=torch.int64).to(target.device))

    loss = 0.

    weights = p_index[:, len(inputs):]
    for i in range(0, len(inputs)):
        r = F.binary_cross_entropy(torch.sigmoid(inputs[i].squeeze()), target.float(), weight=weights[:, i], reduction='mean')
        loss += r

    # x2 = target.clone()
    # x2[target == 1] = 0
    # x2[target == 0] = 1
    # target = -((target - 1) ** 2) + target
    # for i in range(0, len(inputs)):
    #     r = F.margin_ranking_loss(torch.sigmoid(inputs[i].squeeze()), x2.detach(), target, margin=0.6)
    #     loss += r

    return loss / len(inputs)


def Margin_Index_Loss(inputs, target):
    """
    :param input: list[N, N, N....]
    :param target: N
    :return:
    """
    assert torch.equal(torch.unique(target), torch.tensor([0, 1], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([0], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([1], dtype=torch.int64).to(target.device))

    x2 = target.clone()
    x2[target == 1] = 0
    x2[target == 0] = 1
    target = -((target - 1) ** 2) + target
    loss = F.margin_ranking_loss(torch.sigmoid(inputs.squeeze()), x2.detach(), target, margin=0.6)

    return loss


def Margin_Loss_New(inputs, target, p_index=None, type=0):
    assert torch.equal(torch.unique(target), torch.tensor([0, 1], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([0], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([1], dtype=torch.int64).to(target.device))

    loss = 0.

    if type == 0:
        weights = p_index[:, -1]
        a = torch.tensor([0.25, 0.5, 0.75, 1.]).to(target.device)
        for i in range(0, len(inputs)):
            r = F.binary_cross_entropy(torch.sigmoid(inputs[i].squeeze()), target.float(), weight=a[i] * weights,
                                       reduction='mean')
            loss += r

    if type == 1:
        weights = p_index[:, -1]
        for i in range(0, len(inputs)):
            r = F.binary_cross_entropy(torch.sigmoid(inputs[i].squeeze()), target.float(), weight=weights, reduction='mean')
            loss += r

    if type == 2:
        for i in range(0, len(inputs)):
            r = F.binary_cross_entropy(torch.sigmoid(inputs[i].squeeze()), target.float(), reduction='mean')
            loss += r

    return loss/len(inputs)


def Contrastive_loss(label, euclidean_distance, margin=1.0):
    loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance, 2) +
                                  (label)*torch.pow(torch.clamp(margin-euclidean_distance, min=0.0), 2))
    return loss_contrastive


key2loss = {'bce_loss': BCE_Loss}


def get_loss_fun(loss_dict):
    if loss_dict is None:
        return BCE_Loss

    else:
        loss_name = loss_dict["name"]

        if loss_name not in key2loss:
            raise NotImplementedError("{} function not implemented".format(loss_name))
        else:
            loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        return functools.partial(key2loss[loss_name], **loss_params)
