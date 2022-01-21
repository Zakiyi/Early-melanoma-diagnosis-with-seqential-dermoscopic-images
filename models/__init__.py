import torch
from collections import OrderedDict
from .cnn_lstm import CNN_LSTM
from .cnn_pool_s import ResNet_Pool_s
from .cnn_pool_h import ResNet_Pool_h
from .cnn_kaggle import ResNet_Kaggle
from .cnn_hm10001 import ResNet_HM10000
from .conv_diff import TDN
from .conv_diff_p import TDNP
from .conv_diff_h import TDNH
from .conv_diff_hc import TDNHC
from .conv_diff_hc0 import TDNHC0
from .conv_diff_hc_re import TDNHCRE
from .conv_diff_hc_sa import TDNHCSA
from .conv_diff_hc_san import TDNHCSAN
from .conv_diff_hc_sab import TDNHCSAB
from .conv_diff_hc_sac import TDNHCSAC
from .conv_diff_hc_sad import TDNHCSD
from .conv_diff_lstm import TDNH_LSTM

key2model = {
    'cnn-lstm': CNN_LSTM,
    'cnn-pool-s': ResNet_Pool_s,
    'cnn-pool-h': ResNet_Pool_h,
    'cnn-kaggle': ResNet_Kaggle,
    'cnn-hm10000': ResNet_HM10000,
    'cnn-diff-lstm': TDNH_LSTM,
    'cnn-diff-hc0': TDNHC0,
    'cnn-diff-p': TDNP,
    'cnn-diff-h': TDNH,
    'cnn-diff-hc-re': TDNHCRE,
    'cnn-diff-hc-sa': TDNHCSA,
    'cnn-diff-hc-san': TDNHCSAN,
    'cnn-diff-hc-sab': TDNHCSAB,
    'cnn-diff-hc-sac': TDNHCSAC,
    'cnn-diff-hc-sad': TDNHCSD,
    'cnn-diff-hc': TDNHC
}


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


def get_model(model_name, in_channel, n_classes, seq_length, *param):

    if model_name not in key2model.keys():
        print(key2model)
        raise ModelErr('model does not exists in the given list')

    model = key2model[model_name](in_channel, n_classes, seq_length, *param)

    return model


def get_model_output(model, data, modality='MIC', device='cuda', p_index=False):
    # print('inputs: ', inputs.shape)
    if isinstance(data['image'][modality], dict):
        if p_index:
            outputs = model(data['image'][modality]['images'].to(device),
                            data['image'][modality]['diff_images'].to(device),
                            data['image']['p_index'].to(device))  # p_index = N*seq_length
        else:
            print('sss ')
            outputs = model(data['image'][modality]['images'].to(device),
                            data['image'][modality]['diff_images'].to(device))  # p_index = N*seq_length
            print(len(outputs))
    else:
        outputs = model(data['image'][modality].to(device))  # p_index = N*seq_length

    return outputs


class ModelErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
