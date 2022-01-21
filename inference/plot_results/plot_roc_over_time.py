import os
import torch
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from _collections import OrderedDict
from sklearn.metrics import roc_curve, confusion_matrix
from inference.plot_results import mel_ids, ben_ids
reviewers_ben = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
                            'human_results/reviewers_benign.csv')

reviewers_mel = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
                            'human_results/reviewers_malignant.csv')

AI_model_ben = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/'
                           'human_results/AI_model_ben.csv')

AI_model_mel = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/'
                           'CrossValidation_test/human_results/AI_model_mel.csv')


def get_pred_cross_time(preds, padding=True, AI=False):
    if not AI:
        time_1_pred = [p.split('-')[0] for p in preds]
    else:
        time_1_pred = [eval(p)[0] for p in preds]

    time_2_pred = []
    time_3_pred = []
    time_4_pred = []

    for p in preds:
        if not AI:
            p = p.split('-')
        else:
            p = eval(p)
        index = np.linspace(0, len(p)-1, 4, endpoint=True, dtype=np.int)
        if padding:
            time_2_pred.append(p[index[1]])
            time_3_pred.append(p[index[2]])
            time_4_pred.append(p[index[3]])
        else:
            index = index[index != 0]
            if len(p) >= 2:
                time_2_pred.append(p[index[0]])

            if len(p) >= 3:
                time_3_pred.append(p[index[1]])

            if len(p) >= 4:
                time_4_pred.append(p[index[2]])

    return time_1_pred, time_2_pred, time_3_pred, time_4_pred


def compute_metrics(target, pred_label):
    if not np.array_equal(np.unique(pred_label), np.array([0, 1])):
        pred_label[pred_label == 'Benign'] = 0
        pred_label[pred_label == 'Malignant'] = 1
        pred_label[pred_label == 'Melanoma'] = 1

        pred_label[pred_label == 'benign'] = 0
        pred_label[pred_label == 'malignant'] = 1
        pred_label[pred_label == 'melanoma'] = 1
        pred_label = pred_label.astype(np.int)

        # print(pred_label.sum())
    assert np.array_equal(np.unique(pred_label), np.array([0, 1]))

    accuracy = metrics.accuracy_score(target, pred_label)
    tn, fp, fn, tp = confusion_matrix(target, pred_label).ravel()
    specificity = tn / (tn + fp)
    recall = metrics.recall_score(target, pred_label)

    return {'accuracy': accuracy, 'sensitivity': recall, 'specificity': specificity}


def get_pred_from_reviewers():
    ai_ben_predictions = AI_model_ben['diagnose_res']
    time_1_ben_pred, time_2_ben_pred, time_3_ben_pred, time_4_ben_pred = get_pred_cross_time(ai_ben_predictions, AI=True)
    ai_mel_predictions = AI_model_mel['diagnose_res']
    time_1_mel_pred, time_2_mel_pred, time_3_mel_pred, time_4_mel_pred = get_pred_cross_time(ai_mel_predictions, AI=True)

    AI_pred = OrderedDict({'time_1_pred': np.concatenate([np.array(time_1_ben_pred), np.array(time_1_mel_pred)]),
                           'time_1_label': np.concatenate([np.zeros(len(time_1_ben_pred)), np.ones(len(time_1_mel_pred))]),
                           'time_2_pred': np.concatenate([np.array(time_2_ben_pred), np.array(time_2_mel_pred)]),
                           'time_2_label': np.concatenate([np.zeros(len(time_2_ben_pred)), np.ones(len(time_2_mel_pred))]),
                           'time_3_pred': np.concatenate([np.array(time_3_ben_pred), np.array(time_3_mel_pred)]),
                           'time_3_label': np.concatenate([np.zeros(len(time_3_ben_pred)), np.ones(len(time_3_mel_pred))]),
                           'time_4_pred': np.concatenate([np.array(time_4_ben_pred), np.array(time_4_mel_pred)]),
                           'time_4_label': np.concatenate([np.zeros(len(time_4_ben_pred)), np.ones(len(time_4_mel_pred))])
                           })

    AI_metrics = OrderedDict({'time_1': compute_metrics(AI_pred['time_1_label'], AI_pred['time_1_pred']),
                              'time_2': compute_metrics(AI_pred['time_2_label'], AI_pred['time_2_pred']),
                              'time_3': compute_metrics(AI_pred['time_3_label'], AI_pred['time_3_pred']),
                              'time_4': compute_metrics(AI_pred['time_4_label'], AI_pred['time_4_pred'])
                              })

    print(AI_metrics)
    ben_predictions = reviewers_ben['diagnose_result']
    mel_predictions = reviewers_mel['diagnose_result']

    reviewers_pred = OrderedDict({'R1': [], 'R2': [], 'R3': [], 'R4': [], 'R5': [], 'R6': [], 'R7': [], 'R8': [],
                                  'R9': [], 'R10': [], 'R11': [], 'R12': []})

    reviewers_metrics = OrderedDict()
    colors = ['Cadetblue', 'coral', 'chocolate', 'cornflowerblue', 'crimson', 'darkgray', 'darkolivegreen',
              'darkmagenta', 'darkslateblue', 'darkslategray', 'hotpink', 'tan']

    for r, _  in reviewers_pred.items():
        ben_pred = ben_predictions[reviewers_ben['evaluator'] == r]
        time_1_ben_pred, time_2_ben_pred, time_3_ben_pred, time_4_ben_pred = get_pred_cross_time(ben_pred)

        mel_pred = mel_predictions[reviewers_mel['evaluator'] == r]
        time_1_mel_pred, time_2_mel_pred, time_3_mel_pred, time_4_mel_pred = get_pred_cross_time(mel_pred)

        reviewers_pred[r] = OrderedDict({'time_1_pred': np.concatenate([np.array(time_1_ben_pred), np.array(time_1_mel_pred)]),
                                         'time_1_label': np.concatenate([np.zeros(len(time_1_ben_pred)), np.ones(len(time_1_mel_pred))]),
                                         'time_2_pred': np.concatenate([np.array(time_2_ben_pred), np.array(time_2_mel_pred)]),
                                         'time_2_label': np.concatenate([np.zeros(len(time_2_ben_pred)), np.ones(len(time_2_mel_pred))]),
                                         'time_3_pred': np.concatenate([np.array(time_3_ben_pred), np.array(time_3_mel_pred)]),
                                         'time_3_label': np.concatenate([np.zeros(len(time_3_ben_pred)), np.ones(len(time_3_mel_pred))]),
                                         'time_4_pred': np.concatenate([np.array(time_4_ben_pred), np.array(time_4_mel_pred)]),
                                         'time_4_label': np.concatenate([np.zeros(len(time_4_ben_pred)), np.ones(len(time_4_mel_pred))])
                                         })

        # csv = pd.DataFrame({'id': np.concatenate([ben_ids, mel_ids]), 'pred': reviewers_pred[r]['time_4_pred'],
        #                     'label': np.concatenate([np.zeros(len(time_4_ben_pred)), np.ones(len(time_4_mel_pred))])
        #                     })
        # csv.to_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/reviewers_{}.csv'.format(r),
        #            index=False)

        reviewers_metrics[r] = OrderedDict({'time_1': compute_metrics(reviewers_pred[r]['time_1_label'],
                                                                      reviewers_pred[r]['time_1_pred']),
                                            'time_2': compute_metrics(reviewers_pred[r]['time_2_label'],
                                                                      reviewers_pred[r]['time_2_pred']),
                                            'time_3': compute_metrics(reviewers_pred[r]['time_3_label'],
                                                                      reviewers_pred[r]['time_3_pred']),
                                            'time_4': compute_metrics(reviewers_pred[r]['time_4_label'],
                                                                      reviewers_pred[r]['time_4_pred'])
                                            })

        plt.plot(np.arange(4), [reviewers_metrics[r]['time_1']['accuracy'], reviewers_metrics[r]['time_2']['accuracy'],
                                reviewers_metrics[r]['time_3']['accuracy'], reviewers_metrics[r]['time_4']['accuracy']],
                 color=colors[int(r.split('R')[-1])-1])

    plt.plot(np.arange(4), [AI_metrics['time_1']['accuracy'], AI_metrics['time_2']['accuracy'],
                                AI_metrics['time_3']['accuracy'], AI_metrics['time_4']['accuracy']], 'k')
    plt.legend(['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'AI'])
    plt.show()
    return reviewers_metrics


def calculate_roc(pred_score, target):
    fpr, tpr, thresholds = roc_curve(target, pred_score)

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - fpr, index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[roc.tf.argsort()[-1:]]
    roc_t_threshold = list(roc_t['threshold'])[0]

    pred_label = list(map(lambda x: 1 if x > roc_t_threshold else 0, pred_score))

    result = compute_metrics(target, pred_label)
    auc = metrics.roc_auc_score(target, pred_score)

    print('auc ', auc)
    print('acc ', result['accuracy'])
    print('spec ', result['specificity'])
    print('sens ', result['sensitivity'])
    return fpr, tpr, result


reviewers_metrics = get_pred_from_reviewers()
print(reviewers_metrics)

root = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/CST features/SCA/seq_length_4'

T1_res = OrderedDict({'fpr': [], 'tpr': [], 'metrics': []})
T2_res = OrderedDict({'fpr': [], 'tpr': [], 'metrics': []})
T3_res = OrderedDict({'fpr': [], 'tpr': [], 'metrics': []})
T4_res = OrderedDict({'fpr': [], 'tpr': [], 'metrics': []})
matplotlib.style.use('seaborn')
# load prediction score from all folds
for t in [0, 1, 2, 3]:
    scores = []
    targets = []
    for fold in [0, 1, 2, 3, 4]:
        eval_res = torch.load(os.path.join(root, 'fold_{}/exit_scores_eval.pt'.format(fold)))
        scores.extend(eval_res['time_{}'.format(t)]['pred_scores'])
        targets.extend(eval_res['time_{}'.format(t)]['targets'])

    fpr, tpr, result = calculate_roc(scores, targets)

    f, ax = plt.subplots(figsize=(6, 4))
    plt.plot(fpr, tpr)
    AI = plt.scatter(1 - result['specificity'], result['sensitivity'], marker='d', color='k', label='our_model')

    R1 = plt.scatter(1 - reviewers_metrics['R1']['time_{}'.format(t+1)]['specificity'],
                     reviewers_metrics['R1']['time_{}'.format(t+1)]['sensitivity'], marker='s', label='R1')
    R2 = plt.scatter(1 - reviewers_metrics['R2']['time_{}'.format(t+1)]['specificity'],
                     reviewers_metrics['R2']['time_{}'.format(t+1)]['sensitivity'], marker='s', label='R2')
    R3 = plt.scatter(1 - reviewers_metrics['R3']['time_{}'.format(t+1)]['specificity'],
                     reviewers_metrics['R3']['time_{}'.format(t+1)]['sensitivity'], marker='o', label='R3')
    R4 = plt.scatter(1 - reviewers_metrics['R4']['time_{}'.format(t+1)]['specificity'],
                     reviewers_metrics['R4']['time_{}'.format(t+1)]['sensitivity'], marker='s', label='R4')
    R5 = plt.scatter(1 - reviewers_metrics['R5']['time_{}'.format(t+1)]['specificity'],
                     reviewers_metrics['R5']['time_{}'.format(t+1)]['sensitivity'], marker='o', label='R5')
    R6 = plt.scatter(1 - reviewers_metrics['R6']['time_{}'.format(t+1)]['specificity'],
                     reviewers_metrics['R6']['time_{}'.format(t+1)]['sensitivity'], marker='o', label='R6')
    R7 = plt.scatter(1 - reviewers_metrics['R7']['time_{}'.format(t+1)]['specificity'],
                     reviewers_metrics['R7']['time_{}'.format(t+1)]['sensitivity'], marker='s', label='R7')
    R8 = plt.scatter(1 - reviewers_metrics['R8']['time_{}'.format(t+1)]['specificity'],
                     reviewers_metrics['R8']['time_{}'.format(t+1)]['sensitivity'], marker='o', label='R8')
    R9 = plt.scatter(1 - reviewers_metrics['R9']['time_{}'.format(t+1)]['specificity'],
                     reviewers_metrics['R9']['time_{}'.format(t+1)]['sensitivity'], marker='o', label='R9')
    R10 = plt.scatter(1 - reviewers_metrics['R10']['time_{}'.format(t+1)]['specificity'],
                      reviewers_metrics['R10']['time_{}'.format(t+1)]['sensitivity'], marker='o', label='R10')
    R11 = plt.scatter(1 - reviewers_metrics['R11']['time_{}'.format(t+1)]['specificity'],
                      reviewers_metrics['R11']['time_{}'.format(t+1)]['sensitivity'], marker='o', label='R11')
    R12 = plt.scatter(1 - reviewers_metrics['R12']['time_{}'.format(t+1)]['specificity'],
                      reviewers_metrics['R12']['time_{}'.format(t+1)]['sensitivity'], marker='s', label='R12')

    plt.xlabel('1-specificity')
    plt.ylabel('sensitivity')
    plt.legend(handles=[R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, AI])
    plt.tight_layout()
    # plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/CrossValidation_test/human_results/time_{}.png'.format(t), dpi=300)

plt.show()

