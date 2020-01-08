# Copyright (c) 2020 Sophie Giffard-Roisin <sophie.giffard@univ-grenoble-alpes.fr>
# SPDX-License-Identifier: GPL-3.0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

def save_fig_0D(module_name, train_loss, valid_loss, fig_name, log_dir, args):
    # visualize train loss
    plt.title(' type: %s   num_epochs: %s batch_size: %s \n '
              'lr: %s   \n '
              % (module_name, str(args.epochs),
                 str(args.batch_size), str(args.lr)), fontsize=10)

    plt.plot(train_loss, label='train loss', color='green')
    plt.ylabel('loss(mse)')
    plt.xlabel('epochs')

    # visualize validation loss
    plt.plot(valid_loss, label='valid loss', color='red')
    plt.ylabel('validation loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(log_dir + fig_name + '.png')
    plt.savefig(log_dir + fig_name + '.pdf')
    plt.close()


def save_log_0D(module_name, minimum_valid_loss, valid_loss, valid_loss_mae, valid_h_loss, valid_h_loss_mae,
             baseline_losses, baseline_losses_mae, baseline_losses_h, baseline_losses_mae_h, fig_name, log_dir,
             test_loss=None, test_loss_mae=None, test_h_loss=None, test_h_loss_mae=None, args=None):
    log_str = '\n' \
              'figuer name: {fig_name} \n' \
              'module name: {module_name}\n' \
              'optimizer: Adam \t' \
              'learning rate {lr}\t' \
              'betas: {betas}\t' \
              'eps: {eps:.5f}\t' \
              'weight decay: {wd:.5f}\n' \
              'number of epochs: {epochs}\n' \
              'minimum valid loss: {minimum_valid_loss:.5f}\n' \
              'valid_loss: {valid_losses:.5f} \t' \
              'valid_loss_mae: {valid_losses_mae:.5f} \n' \
              'valid_hurricanes_loss: {valid_h_losses:.5f} \t' \
              'valid_hurricanes_loss_mae: {valid_h_losses_mae:.5f} \n' \
              'baseline_loss {baseline_losses:.5f} \t' \
              'baseline_loss_mae {baseline_losses_mae:.5f}\n' \
              'baseline_loss_hurricanes {baseline_losses_h:.5f} \t' \
              'baseline_loss_mae_hurricanes {baseline_losses_mae_h:.5f}\n' \
        .format(fig_name=fig_name, module_name=module_name,
                lr=str(args.lr), betas=str(args.betas),
                eps=args.eps, wd=args.weight_decay,
                epochs=str(args.epochs),
                minimum_valid_loss=minimum_valid_loss, valid_losses=valid_loss, valid_losses_mae=valid_loss_mae,
                valid_h_losses=valid_h_loss, valid_h_losses_mae=valid_h_loss_mae,
                baseline_losses=baseline_losses, baseline_losses_mae=baseline_losses_mae,
                baseline_losses_h=baseline_losses_h, baseline_losses_mae_h=baseline_losses_mae_h)
    if test_loss:
        log_str = log_str + '\n \n (test_loss: {test_losses:.5f} \t' \
                            'test_loss_mae: {test_losses_mae:.5f}) \n' \
                            '(test_hurricanes_loss: {test_h_losses:.5f} \t' \
                            'test_hurricanes_loss_mae: {test_h_losses_mae:.5f}) \n' \
                            .format(test_losses=test_loss, test_losses_mae = test_loss_mae,
                                    test_h_losses=test_h_loss, test_h_losses_mae=test_h_loss_mae)
    with open(log_dir + 'log.txt', 'a+') as f:
        f.write(log_str)


def save_forecast_result_0D(model, testset, testloader,log_dir, num_tracks=2, name='result'):
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(testloader):
            # add windspeed and J-day
            input = target[:,-(num_tracks+2):-1,:].view(target.shape[0],-1)
            # add long/lat
            longlat = target[:,0,:].view(target.shape[0],-1)
            # add dist2land
            dist2land = target[:,-1,0].view(target.shape[0],-1)
            input = torch.cat((input, longlat), 1)
            input = torch.cat((input, dist2land), 1)
            if i == 0:
                output = model(input)
            else:
                output_ = model(input)
                output = torch.cat((output, output_), dim=0)

    predicted = np.array(output)
    loc=testset.labels[:,0,:]
    ground_truth = testset.labels[:,1,:]
    ids = testset.ids
    timestep = testset.timestep
    predicted = np.concatenate((predicted, ids.reshape(-1,1)), axis=1)
    predicted = np.concatenate((predicted, np.array(timestep).reshape(-1,1)), axis=1)
    predicted = np.concatenate((predicted, loc, ground_truth), axis=1)
    np.savetxt(log_dir + name+ ".csv", predicted, delimiter=",",fmt='%s, %s, %s, %s, %s, %s, %s, %s ')


def save_fig_2D(module_name, train_loss, valid_loss, fig_name, log_dir, args):
    # visualize training loss
    plt.title(' type: %s   num_epochs: %s batch_size: %s \n '
              'lr: %s   \n '
              % (module_name, str(args.epochs_0) + "+" + str(args.epochs_1) + "+" + str(args.epochs_2),
                 str(args.batch_size), str(args.lr_0) + '+' + str(args.lr_1) + '+' + str(args.lr_2)), fontsize=10)

    plt.plot(train_loss, label='train loss', color='green')
    plt.ylabel('loss(mse)')
    plt.xlabel('epochs')

    # visualize validation loss
    plt.plot(valid_loss, label='valid loss', color='red')
    plt.ylabel('validation loss')
    plt.xlabel('epochs')

    plt.legend()
    plt.savefig(log_dir + fig_name + '.png')
    plt.savefig(log_dir + fig_name + '.pdf')
    plt.close()


def save_log_2D(module_name, minimum_valid_loss, valid_loss, valid_loss_mae, valid_h_loss, valid_h_loss_mae,
             baseline_losses, baseline_losses_mae, baseline_losses_h, baseline_losses_mae_h, fig_name, log_dir,
             test_loss=None, test_loss_mae=None, test_h_loss=None, test_h_loss_mae=None, args=None):
    log_str = '\n' \
              'figure name: {fig_name} \n' \
              'module name: {module_name}\n' \
              'optimizer: Adam \t' \
              'learning rate {lr}\t' \
              'betas: {betas}\t' \
              'eps: {eps:.5f}\t' \
              'weight decay: {wd:.5f}\n' \
              'dropout rate: {dropout:.5f}\n' \
              'number of epochs: {epochs}\n' \
              'minimum valid loss: {minimum_valid_loss:.5f}\n' \
              'valid_loss: {valid_losses:.5f} \t' \
              'valid_loss_mae: {valid_losses_mae:.5f} \n' \
              'valid_hurricanes_loss: {valid_h_losses:.5f} \t' \
              'valid_hurricanes_loss_mae: {valid_h_losses_mae:.5f} \n' \
              'baseline_loss {baseline_losses:.5f} \t' \
              'baseline_loss_mae {baseline_losses_mae:.5f}\n' \
              'baseline_loss_hurricanes {baseline_losses_h:.5f} \t' \
              'baseline_loss_mae_hurricanes {baseline_losses_mae_h:.5f}\n' \
        .format(fig_name=fig_name, module_name=module_name,
                lr=str(args.lr_0) + '+' + str(args.lr_1) + '+' + str(args.lr_2), betas=str(args.betas),
                eps=args.eps, wd=args.weight_decay, dropout=args.dropout,
                epochs=str(args.epochs_0) + "+" + str(args.epochs_1) + "+" + str(args.epochs_2),
                minimum_valid_loss=minimum_valid_loss, valid_losses=valid_loss, valid_losses_mae=valid_loss_mae,
                valid_h_losses=valid_h_loss, valid_h_losses_mae=valid_h_loss_mae,
                baseline_losses=baseline_losses, baseline_losses_mae=baseline_losses_mae,
                baseline_losses_h=baseline_losses_h, baseline_losses_mae_h=baseline_losses_mae_h)

    if test_loss:
        log_str = log_str + '\n \n (test_loss: {test_losses:.5f} \t' \
                            'test_loss_mae: {test_losses_mae:.5f}) \n' \
                            '(test_hurricanes_loss: {test_h_losses:.5f} \t' \
                            'test_hurricanes_loss_mae: {test_h_losses_mae:.5f}) \n' \
                            .format(test_losses = test_loss, test_losses_mae = test_loss_mae,
                                    test_h_losses = test_h_loss, test_h_losses_mae=test_h_loss_mae)

    with open(log_dir + 'log.txt', 'a+') as f:
        f.write(log_str)
