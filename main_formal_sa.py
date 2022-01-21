import os
import time
import json
import yaml
import math
import torch
import random
import numpy as np
import pickle as pkl
import torch.optim as optim
from sklearn import metrics
from models import get_model
from models import get_model_output
from misc import plot_progress
import torch.nn.functional as F
from data_proc.sequence_aug_diff import Augmentations_diff
from data_proc.ssd_datasplit import ssd_split
from data_proc.ssd_dataset import Skin_Dataset
from torch.utils.data import DataLoader
from misc.training_manager import training_manager
from collections import OrderedDict
from misc.utils import get_pred_class_index
from misc.loss_function import get_loss_fun, BCE_Index_Loss, BCE_Loss

threshold = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
rank = []


def Self_Distillation(inputs, target=None, teacher_out=None, p_index=None, alpha=0.4):
    """
    loss function for Knowledge Distillation (KD)
    outputs: list[N, N, N]
    labels: N

    """
    loss = 0
    assert len(inputs) > 1
    weights = p_index[:, len(inputs):]

    for i in range(len(inputs) - 1):
        l = F.kl_div(torch.log(torch.sigmoid(inputs[i])), torch.sigmoid(inputs[-1]).detach(), reduction='batchmean')
        b = (1. - alpha) * F.binary_cross_entropy(torch.sigmoid(inputs[i].squeeze()), target.float(), reduction='mean') + alpha * l

        loss += b

    dst_last = F.binary_cross_entropy(torch.sigmoid(inputs[-1].squeeze()), target.float(), reduction='mean')

    loss = loss + dst_last

    return loss / len(inputs)


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

    return loss / len(inputs)


def set_seed(seed=2019):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train(cfg, model, train_dataloader, optimizer, loss_fn, epoch=0, teacher_model=None, m=(0.1, -0.1)):
    loss_group = OrderedDict({'img_loss': [], 'diff_loss': [], 'avg_loss': [], 'rank_loss': [], 'margin_loss': [], 'distill_loss': []})
    model.train()
    train_epoch_results = OrderedDict({'train_loss': [],
                                       'target': [],
                                       'pred_score': [],
                                       'pred_label': []})

    torch.autograd.set_detect_anomaly(True)
    for data in train_dataloader:
        optimizer.zero_grad()
        outputs, spatial_scores, temporal_scores, exit_scores = get_model_output(model, data,
                                                                    cfg['data']['modality'],
                                                                    device, cfg['data']['p_index'])

        if isinstance(outputs, list) or isinstance(outputs, tuple):
            if len(outputs) > 1:
                img_loss = BCE_Loss(outputs[0], data['target'].to(device))
                diff_loss = BCE_Loss(outputs[1], data['target'].to(device))
                # avg_loss = BCE_Loss(outputs[-1], data['target'].to(device))

                avg_loss = BCE_Index_Loss(outputs[-1], data['target'].to(device), p_index=data['image']['p_index'].to(device))

                distill_loss = Self_Distillation(exit_scores, data['target'].to(device), p_index=data['image']['p_index'].to(device))
                margin_loss = Margin_Loss(exit_scores, data['target'].to(device), p_index=data['image']['p_index'].to(device))

                loss = m[0]*avg_loss + m[1]*distill_loss
                # loss = m[2] * rank_loss + m[4] * margin_loss
                # loss = avg_loss
                print('the curent weights of two loss is rank_distill_{}_{}'.format(m[0], m[1]))
                # print('we are using rank margin of {}_{}'.format(m[0], m[1]))
                # print('current a is {:.3f}'.format(a))
                print('main loss: ', avg_loss)
                print('distill loss: ', distill_loss)
                # print('rank loss: ', rank_loss)
                print('margin loss: ', margin_loss)
                print('loss: ', loss)
                print('\n')
            else:
                loss = loss_fn(outputs[-1], data['target'].to(device))

            # pred_score = outputs[-1].detach().cpu().numpy()
            pred_score = outputs[-1].detach().cpu().numpy()
        else:
            loss = loss_fn(outputs, data['target'].to(device))
            pred_score = outputs.detach().cpu().numpy()

        loss.backward()
        optimizer.step()

        "change to exit score as prediction"
        pred_score = exit_scores[-1].sigmoid().squeeze().detach().cpu().numpy()

        # TODO: checking the pred_score and pred_label format
        train_epoch_results['train_loss'].append(loss.item())
        train_epoch_results['target'] += list(data['target'].detach().cpu().numpy())
        train_epoch_results['pred_score'] += list(pred_score.squeeze())
        train_epoch_results['pred_label'] += list(get_pred_class_index(pred_score, threshold))
        loss_group['img_loss'].append(img_loss.item())
        loss_group['diff_loss'].append(diff_loss.item())
        loss_group['avg_loss'].append(avg_loss.item())
        # loss_group['rank_loss'].append(rank_loss.item())
        loss_group['margin_loss'].append(margin_loss.item())
        loss_group['distill_loss'].append(distill_loss.item())
    return train_epoch_results, loss_group


def val(cfg, model, val_dataloader, loss_fn, epoch=0, teacher_model=None, m=(0.1, -0.1)):
    val_eval_results = OrderedDict({'val_loss': [], 'target': [], 'pred_score': [], 'pred_label': []})
    loss_group = OrderedDict({'img_loss': [], 'diff_loss': [], 'avg_loss': [], 'rank_loss': [], 'margin_loss': [], 'distill_loss': []})
    model.eval()

    with torch.no_grad():
        for data in val_dataloader:
            outputs, spatial_scores, temporal_scores, exit_scores = get_model_output(model, data,
                                                                        cfg['data']['modality'],
                                                                        device, cfg['data']['p_index'])  # N*C
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                if len(outputs) > 1:
                    img_loss = BCE_Loss(outputs[0], data['target'].to(device))
                    diff_loss = BCE_Loss(outputs[1], data['target'].to(device))
                    # avg_loss = BCE_Loss(outputs[-1], data['target'].to(device))

                    avg_loss = BCE_Index_Loss(outputs[-1], data['target'].to(device), p_index=data['image']['p_index'].to(device))
                    margin_loss = Margin_Loss(exit_scores, data['target'].to(device), p_index=data['image']['p_index'].to(device))
                    # rank_loss = Ranking_Loss(spatial_scores, data['target'].to(device), margin=(0.05, 0.00), p_index=data['image']['p_index'].to(device))
                    distill_loss = Self_Distillation(exit_scores, data['target'].to(device), p_index=data['image']['p_index'].to(device))

                    val_loss = m[0]*avg_loss + m[1]*distill_loss
                else:
                    val_loss = loss_fn(outputs[0], data['target'].to(device))

                # pred_score = outputs[-1].detach().cpu().numpy()
                pred_score = outputs[-1].detach().cpu().numpy()

            else:
                val_loss = loss_fn(outputs, data['target'].to(device))
                pred_score = outputs.detach().cpu().numpy()

            "change to exit score as prediction"
            pred_score = exit_scores[-1].sigmoid().squeeze().detach().cpu().numpy()

            val_eval_results['val_loss'].append(val_loss.item())
            val_eval_results['target'] += list(data['target'].detach().cpu().numpy())
            val_eval_results['pred_score'] += list(pred_score.squeeze())
            val_eval_results['pred_label'] += list(get_pred_class_index(pred_score, threshold))

            loss_group['img_loss'].append(img_loss.item())
            loss_group['diff_loss'].append(diff_loss.item())
            loss_group['avg_loss'].append(avg_loss.item())
            # loss_group['rank_loss'].append(rank_loss.item())
            loss_group['margin_loss'].append(margin_loss.item())
            loss_group['distill_loss'].append(distill_loss.item())
    return val_eval_results, loss_group


def ten_crop_test(cfg, model, dataset):
    test_results = OrderedDict({'target': [], 'pred_score': [], 'pred_label': []})
    model.eval()
    with torch.no_grad():
        for data in iter(dataset):
            outputs, _, __, e = get_model_output(model, data, cfg['data']['modality'], device, cfg['data']['p_index'])

            if isinstance(outputs, list) or isinstance(outputs, tuple):
                pred_score = e[-1].detach().cpu().numpy()
            else:
                pred_score = outputs.detach().cpu().numpy()     # Ncrops * 1

            pred_score = np.mean(pred_score, axis=0)

            test_results['target'] += [data['target'].detach().cpu().numpy()]
            test_results['pred_score'] += [pred_score.squeeze()]
            test_results['pred_label'] += [get_pred_class_index(pred_score, threshold)]

    accuracy = metrics.accuracy_score(test_results['target'], test_results['pred_label'])
    auc = metrics.roc_auc_score(test_results['target'], test_results['pred_score'])
    recall = metrics.recall_score(test_results['target'], test_results['pred_label'])
    precision = metrics.precision_score(test_results['target'], test_results['pred_label'])
    f1_score = metrics.f1_score(test_results['target'], test_results['pred_label'])

    return {'accuracy': accuracy, 'auc': auc, 'recall': recall, 'precision': precision, 'f1-score': f1_score}


def main(cfg, m):

    """step 1: setup data"""
    data_root = cfg['data']['root']
    run_exp_dir = cfg['run_exp']

    if not os.path.exists(cfg['training']['result_dir']):
        os.makedirs(cfg['training']['result_dir'])

    if not os.path.exists(cfg['data']['k_split']):
        ssd_split(data_root, run_exp_dir).data_split()

    with open(os.path.join(run_exp_dir, 'data_setting', 'data_split.pkl'), 'rb') as f:
        data_split_file = pkl.load(f)

    train_aug_parameters = OrderedDict({'affine': None,
                                        'flip': True,
                                        'color_trans': {'brightness': (0.8, 1.2),
                                                        'contrast': (0.8, 1.2),
                                                        'saturation': (0.8, 1.2),
                                                        'hue': (-0.03, 0.03)},
                                        'normalization': {'mean': (0.485, 0.456, 0.406),
                                                          'std': (0.229, 0.224, 0.225)},
                                        'size': 320,
                                        'scale': (0.8, 1.2),
                                        'ratio': (0.9, 1.1)
                                        }
                                       )
    val_aug_parameters = OrderedDict({'affine': None,
                                      'flip': True,
                                      'color_trans': {'brightness': (1.0, 1.0),
                                                      'contrast': (1.0, 1.0),
                                                      'saturation': (1.0, 1.0),
                                                      'hue': (-0.001, 0.001)},
                                      'normalization': {'mean': (0.485, 0.456, 0.406),
                                                        'std': (0.229, 0.224, 0.225)},
                                      'size': 320,
                                      'scale': (1.0, 1.0),
                                      'ratio': (1.0, 1.0)
                                      }
                                     )
    train_augmentor = Augmentations_diff(train_aug_parameters, color_recalibration=True, padding_mode='negative')
    val_augmentor = Augmentations_diff(val_aug_parameters, color_recalibration=True, padding_mode='negative')

    train_dataset = Skin_Dataset(data_root, data_split_file[cfg['fold']]['train'], seq_length=cfg['data']['seq_length'],
                                 transform=train_augmentor.transform, data_modality=cfg['data']['modality'], is_train=True)

    val_dataset = Skin_Dataset(data_root, data_split_file[cfg['fold']]['val'], seq_length=cfg['data']['seq_length'],
                               transform=val_augmentor.transform, data_modality=cfg['data']['modality'], is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
                                  num_workers=cfg['training']['num_workers'], drop_last=True)

    val_dataloader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False,
                                num_workers=cfg['training']['num_workers'], drop_last=True)

    """step 2: setup model & loss_fn & optimizer"""
    assert cfg['training']['optimizer']['name'] == 'Adam'
    assert cfg['training']['lr_scheduler']['name'] == 'ReduceLROnPlateau'
    train_loss_group = OrderedDict({'img_loss': [], 'diff_loss': [], 'avg_loss': [], 'rank_loss': [], 'margin_loss': [], 'distill_loss': []})
    val_loss_group = OrderedDict({'img_loss': [], 'diff_loss': [], 'avg_loss': [], 'rank_loss': [], 'margin_loss': [], 'distill_loss': []})

    train_manager = training_manager(cfg['training']['result_dir'], cfg['model'], optim.Adam, optim.lr_scheduler.ReduceLROnPlateau, device,
                                     cfg['training']['max_epoch'])

    train_manager.initialize_model(cfg['data']['channel'], train_dataset.n_class, cfg['data']['seq_length'],
                                   cfg['training']['learning_rate'], cfg['training']['optimizer'],
                                   cfg['training']['lr_scheduler'], device, cfg['training']['fine_tune'])

    loss_fn = get_loss_fun(cfg['training']['loss'])
    train_manager.print_to_log_file('we are using ' + cfg['training']['loss']['name'])

    if os.path.exists(cfg['training']['resume_path']):
        train_manager.load_checkpoint(best_model=False)

    with open(os.path.join(train_manager.output_dir, 'configs.yml'), 'w') as config_file:
        yaml.dump(cfg, config_file, default_flow_style=False)

    while train_manager.epoch < train_manager.max_epoch:
        train_manager.print_to_log_file("\nepoch: ", train_manager.epoch)
        epoch_start_time = time.time()

        train_epoch_results, train_loss = train(cfg, train_manager.model, train_dataloader, train_manager.optimizer,
                                                loss_fn, train_manager.epoch,  m=m)
        val_epoch_results, val_loss = val(cfg, train_manager.model, val_dataloader, loss_fn, train_manager.epoch,
                                          m=m)

        # evaluation, plot_training_curve, update_lr, save_checkpoint, update eval_criterion
        train_manager.train_losses.append(np.mean(train_epoch_results['train_loss']))
        train_manager.val_losses.append(np.mean(val_epoch_results['val_loss']))
        #
        print('train epoch loss {:.4f}'.format(train_manager.train_losses[-1]))
        print('val epoch loss {:.4f}'.format(train_manager.val_losses[-1]))

        train_manager.compute_metrics(train_epoch_results, train=True)
        train_manager.compute_metrics(val_epoch_results, train=False)
        print('train_auc {:.4f}'.format(train_manager.train_eval_metrics['auc'][-1]))
        print('val_auc {:.4f}'.format(train_manager.val_eval_metrics['auc'][-1]))

        train_manager.epoch += 1
        continue_training = train_manager.run_on_epoch_end()

        epoch_end_time = time.time()
        if not continue_training:
            break

        train_manager.print_to_log_file("This epoch took {:.2f} s\n".format(epoch_end_time - epoch_start_time))

        train_loss_group['img_loss'].append(np.mean(train_loss['img_loss']))
        train_loss_group['diff_loss'].append(np.mean(train_loss['diff_loss']))
        train_loss_group['avg_loss'].append(np.mean(train_loss['avg_loss']))
        train_loss_group['rank_loss'].append(np.mean(train_loss['rank_loss']))
        train_loss_group['margin_loss'].append(np.mean(train_loss['margin_loss']))
        train_loss_group['distill_loss'].append(np.mean(train_loss['distill_loss']))

        val_loss_group['img_loss'].append(np.mean(val_loss['img_loss']))
        val_loss_group['diff_loss'].append(np.mean(val_loss['diff_loss']))
        val_loss_group['avg_loss'].append(np.mean(val_loss['avg_loss']))
        val_loss_group['rank_loss'].append(np.mean(val_loss['rank_loss']))
        val_loss_group['margin_loss'].append(np.mean(val_loss['margin_loss']))
        val_loss_group['distill_loss'].append(np.mean(val_loss['distill_loss']))
        plot_progress(train_loss_group, val_loss_group, train_manager.output_dir)

    train_manager.save_checkpoint(os.path.join(train_manager.output_dir, train_manager.model_name + "_final.model"))
    # now we can delete latest as it will be identical with final
    if os.path.isfile(os.path.join(train_manager.output_dir, train_manager.model_name + "_scheduled.model")):
        os.remove(os.path.join(train_manager.output_dir, train_manager.model_name + "_scheduled.model"))

    if cfg['ten_crop_test']:
        test_aug_parameters = OrderedDict({'affine': None,
                                           'flip': True,
                                           'color_trans': None,
                                           'normalization': {'mean': (0.485, 0.456, 0.406),
                                                             'std': (0.229, 0.224, 0.225)},
                                           'size': 320,
                                           'scale': (0.9, 1.1),
                                           'ratio': (0.9, 1.1)
                                           }
                                          )

        test_augmentor = Augmentations_diff(test_aug_parameters, test_mode=True, color_recalibration=True, padding_mode='negative')
        test_dataset = Skin_Dataset(data_root, data_split_file[cfg['fold']]['val'], seq_length=cfg['data']['seq_length'],
                                    transform=test_augmentor.transform, data_modality=cfg['data']['modality'],
                                    is_train=False, test_mode=True)

        saved_model = torch.load(os.path.join(train_manager.output_dir, train_manager.model_name + "_best.model"))
        train_manager.model.load_state_dict(saved_model['model_state_dict'])
        result_metrics = ten_crop_test(cfg, train_manager.model, test_dataset)
        with open(os.path.join(cfg['training']['result_dir'], 'ten_crop_result.json'), 'w') as res:
            json.dump(result_metrics, res, indent=4, sort_keys=True)

        r = np.array(rank)
        np.save(os.path.join(cfg['training']['result_dir'], 'rank.mat'), r)


if __name__ == '__main__':
    seed = random.randint(1, 10000)
    yml_file = '/configs/skin_config.yml'
    teacher_models = '/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/cnn-diff-hc-sa_MIC/baseline_two_stream'
    with open(yml_file) as f:
        config_file = yaml.load(f)

    assert config_file['fold'] > 0
    # config_file['training']['seed'] = 7024

    if config_file['debug']:
        config_file['training']['max_epoch'] = 1

    for round in [0, 1, 2, 3, 4]:
        for m in [(0.8, 0.2)]:
            for seq_length in [4]:
                config_file['data']['seq_length'] = seq_length
                config_file['data']['p_index'] = False
                for fold in [3, 2, 4, 0, 1]:
                    config_file['fold'] = fold
                    config_file['training']['teacher'] = os.path.join(teacher_models, 'seq_length_' + str(seq_length),
                                                                      'fold_' + str(fold), config_file['model'] + '_best.model')
                    config_file['model'] = 'cnn-diff-hc-sa'
                    # print(config_file['training']['fine_tune']['pre_trained_model'])
                    if not os.path.exists(config_file['training']['resume_path']):
                        run_dir = config_file['model'] + '_' + config_file['data']['modality']
                        config_file['training']['result_dir'] = os.path.join('/home/zyi/My_disk/medical_AI/cnn-diff-hc-sa',
                                                                             'joint_train_scratch', 'round_{}'.format(round),
                                                                             'seq_length_{}'.format(
                                                                                 config_file['data']['seq_length']),
                                                                             'fold_{}'.format(config_file['fold']))
                    else:
                        config_file['training']['result_dir'] = config_file['training']['resume_path']

                    main(config_file, m)