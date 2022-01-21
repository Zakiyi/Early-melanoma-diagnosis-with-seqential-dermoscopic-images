import os
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def plot_pred(pos_preds, neg_preds, output_dir):
    try:
        font = {'weight': 'normal', 'size': 14}

        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        pos_preds = pos_preds.detach().cpu().numpy()
        neg_preds = neg_preds.detach().cpu().numpy()

        ax.plot(list(range(pos_preds.shape[-1])), pos_preds.mean(axis=0), color='coral', ls='-', label='melanoma')
        ax.plot(list(range(neg_preds.shape[-1])), neg_preds.mean(axis=0), color='g', ls='-', label='benign')

        ax.fill_between(list(range(pos_preds.shape[-1])), pos_preds.mean(axis=0) - pos_preds.std(axis=0),
                        pos_preds.mean(axis=0) + pos_preds.std(axis=0), color='coral', alpha=0.18)

        ax.fill_between(list(range(neg_preds.shape[-1])), neg_preds.mean(axis=0) - neg_preds.std(axis=0),
                        neg_preds.mean(axis=0) + neg_preds.std(axis=0), color='g', alpha=0.18)
        # for i in range(pos_preds.shape[0]):
        #     pos_pred = pos_preds[i, :]
        #     ax.plot(list(range(pos_preds.shape[-1])), pos_pred, color='coral', ls='-')
        #
        # for i in range(neg_preds.shape[0]):
        #     neg_pred = neg_preds[i, :]
        #     ax.plot(list(range(neg_preds.shape[-1])), neg_pred, color='b', ls='-')

        ax.set_xlabel("time steps")
        ax.set_ylabel("pred scores")
        ax.legend()

        fig.savefig(output_dir)
        plt.close()

    except IOError:
        raise IOError


def plot_loss(train_loss, val_loss, output_dir):
    try:
        font = {'weight': 'normal', 'size': 12}

        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax.plot(list(range(len(train_loss))), train_loss, color='coral', ls='-', label="loss_train")

        ax.plot(list(range(len(val_loss))), val_loss, color='b', ls='-', label="loss_val")

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()

        fig.savefig(output_dir)
        plt.close()

    except IOError:
        raise IOError


def plot_progress(train_losses, val_losses, output_dir):
    """

    :param train_losses: e.g. {'img_loss': [], 'diff_loss': [], 'avg_loss': []}
    :param val_losses:
    :param output_dir:
    :return:
    """
    # for name, value in train_losses.items():
    #     plot_loss(train_losses[name], val_losses[name], os.path.join(output_dir, "{}_moniter.png".format(name)))

    plot_loss(train_losses['img_loss'], val_losses['img_loss'], os.path.join(output_dir, "img_loss_moniter.png"))
    plot_loss(train_losses['diff_loss'], val_losses['diff_loss'], os.path.join(output_dir, "diff_loss_moniter.png"))
    plot_loss(train_losses['avg_loss'], val_losses['avg_loss'], os.path.join(output_dir, "avg_loss_moniter.png"))

    try:
        plot_loss(train_losses['rank_loss'], val_losses['rank_loss'], os.path.join(output_dir, "rank_loss_moniter.png"))
    except KeyError:
        pass

    try:
        plot_loss(train_losses['margin_loss'], val_losses['margin_loss'], os.path.join(output_dir, "margin_loss_moniter.png"))
    except KeyError:
        pass

    try:
        plot_loss(train_losses['distill_loss'], val_losses['distill_loss'], os.path.join(output_dir, "distill_loss_moniter.png"))
    except KeyError:
        pass