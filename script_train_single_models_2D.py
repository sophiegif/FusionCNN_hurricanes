import sys
#sys.path.append('/home/tau/myang/ClimateSaclayRepo/')
import os
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import Model_uv, Model_z

from Utils.loss import Regress_Loss, Regress_Loss_Mae
from Utils.Utils import save_checkpoint, early_stopping, dataset_filter_3D, extract_hurricanes, get_baselineset
from Utils.funcs import *
from Utils.model_inputs_3D import load_datasets
from Utils.save_figs_logs import save_log_2D, save_fig_2D

######################
# hyper-parameters to set here
######################

data_dir = "/data/titanic_1/users/sophia/myang/model/data_3d_uvz_historic12h/"
log_dir = "/data/titanic_1/users/sophia/myang/logs/3conv_models_24h/"

levels = (2, 5, 7)
params = (0,1,2,3,4,5,6,7,8)
uv_params = (0,1,3,4,6,7)
z_params = (2,5,8)
# params: 0:u 1:v, 2:z
load_t = (0, -6)
arch = "Net_2d_conv3"

######################
# parser
######################


def launch_parser():
    parser = argparse.ArgumentParser(
        description='training parameters')
    subparsers = parser.add_subparsers(dest='category')
    subparsers.required = True
    parser_category = subparsers.add_parser(
        'category', description='predict category')
    parser_displacement = subparsers.add_parser(
        'displacement', description='predict displacement')
    parsers = [parser_category, parser_displacement]

    predict_hours = (6, 24)
    num_tracks = (0, 1, 2, 3, 4)
    for pr in parsers:
        pr.add_argument('--hours', metavar='HOURS', default=24,
                        type=int, choices=predict_hours,
                        help='prediction hours: ' + ' (default: 24)')
        pr.add_argument('--num_tracks', default=2, type=int, metavar='NT',
                        choices=num_tracks, help='number of tracks in y label')
        pr.add_argument('--epochs_0', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
        pr.add_argument('--epochs_1', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
        pr.add_argument('--epochs_2', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
        pr.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
        pr.add_argument('--lr_0', '--learning-rate-0', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
        pr.add_argument('--lr_1', '--learning-rate-1', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
        pr.add_argument('--lr_2', '--learning-rate-2', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate for second model')
        pr.add_argument('--betas', default=(0.9, 0.999), type=float, metavar='B',
                        help='betas')
        pr.add_argument('--eps', default=1e-8, type=float, metavar='EPS',
                        help='eps')
        pr.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
        pr.add_argument('--dropout', '--do', default=0.0, type=float,
                        metavar='DO', help='drop out rate (default: 0)')
        pr.add_argument('--save_fig_name', default='0', type=str, metavar='N',
                        help='give the name of saved fig and log')
    arguments = parser.parse_args()

    return arguments

##############################


def main(log_dir, validset, testset, trainloader, validloader, testloader, validloader_h,
             validloader_baseline, validloader_baseline_h, args):
    best_score = 1e10
    fig_name = args.save_fig_name
    if os.path.isdir(log_dir+'models'+'/'):
        pass
    else:
        os.mkdir(log_dir+'models'+'/')
    model_dir = log_dir+'models'+'/'

    criterion = Regress_Loss()
    criterion_mae = Regress_Loss_Mae()

    print('--------------------------uv model training---------------------------')

    model_uv = Model_uv.__dict__[arch](dropout=args.dropout, levellist=levels, params=uv_params).double().cuda()
    model_uv = nn.DataParallel(model_uv).cuda()

    optimizer = optim.Adam(model_uv.parameters(), lr=args.lr_0, betas=args.betas, eps=args.eps,
                           weight_decay=args.weight_decay)
    train_loss = []
    valid_loss = []
    test_loss = []
    for epoch in range(0, args.epochs_0):
        print('start training with epoch(%d/%5d)' % (epoch + 1, args.epochs_0))

        # train for one epoch
        train_2D(trainloader, model_uv, criterion, optimizer, epoch)

        # validate for the epoch
        train_losses, train_losses_mae = get_score_2D(trainloader, model_uv, criterion, criterion_mae)
        val_losses, val_losses_mae = get_score_2D(validloader, model_uv, criterion, criterion_mae)
        test_losses, test_losses_mae = get_score_2D(testloader, model_uv, criterion, criterion_mae)

        print('valid loss:', val_losses.avg)
        print('test loss:', test_losses.avg)
        score = val_losses.avg
        is_best = score < best_score  # smaller the better
        best_score = min(score, best_score)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_uv.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=model_dir , name='uv_model_best.pth.tar')
        valid_loss.append(val_losses.avg)
        train_loss.append(train_losses.avg)
        test_loss.append(test_losses.avg)

    if os.path.isfile(model_dir  + 'uv_model_best.pth.tar'):
        checkpoint = torch.load(model_dir + 'uv_model_best.pth.tar')
    else:
        raise ValueError('could not find checkpoint')
    save_fig_2D(model_uv.__class__.__name__, train_loss, valid_loss, fig_name + '_uv_alone', log_dir, args)

    print('------------------uv model testing----------------------------')

    model_uv.load_state_dict(checkpoint['state_dict'])
    minimum_valid_loss = checkpoint['best_score']
    baseline_losses, baseline_losses_mae = get_baseline_score(validloader_baseline, criterion, criterion_mae,
                                                              hours=args.hours)
    baseline_losses_h, baseline_losses_mae_h = get_baseline_score(validloader_baseline_h, criterion, criterion_mae,
                                                                  hours=args.hours)
    valid_losses, valid_losses_mae = get_score_2D(validloader, model_uv, criterion, criterion_mae)
    valid_h_losses, valid_h_losses_mae = get_score_2D(validloader_h, model_uv, criterion, criterion_mae)

    save_forecast_result(model_uv, validset,validloader, log_dir, num_tracks=args.num_tracks, name='results_uv_valid')
    save_forecast_result(model_uv, testset,testloader, log_dir, num_tracks=args.num_tracks, name='results_uv_test')
    save_log_2D(model_uv.__class__.__name__, minimum_valid_loss, valid_losses.avg, valid_losses_mae.avg, valid_h_losses.avg,
             valid_h_losses_mae.avg,
             baseline_losses.avg, baseline_losses_mae.avg, baseline_losses_h.avg, baseline_losses_mae_h.avg,
             fig_name + '_uv_alone', log_dir, args)

    print('------------------z model training----------------------------')

    best_score = 1e10
    model_z = Model_z.__dict__[arch](dropout=args.dropout, levellist=levels, params=z_params).double().cuda()
    module_name = model_z.__class__.__name__
    model_z = nn.DataParallel(model_z).cuda()
    optimizer = optim.Adam(model_z.parameters(), lr=args.lr_1, betas=args.betas, eps=args.eps,
                           weight_decay=args.weight_decay)
    cudnn.benchmark = True

    train_loss = []
    valid_loss = []
    for epoch in range(0, args.epochs_1):
        print('start training with epoch(%d/%5d)' % (epoch + 1, args.epochs_1))

        # train for one epoch
        train_2D(trainloader, model_z, criterion, optimizer, epoch)

        # validate for the epoch
        train_losses, train_losses_mae = get_score_2D(trainloader, model_z, criterion, criterion_mae)
        val_losses, val_losses_mae = get_score_2D(validloader, model_z, criterion, criterion_mae)

        print('valid loss:', val_losses.avg)
        score = val_losses.avg
        is_best = score < best_score  # smaller the better
        best_score = min(score, best_score)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_z.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=model_dir, name='z_model_best.pth.tar')
        valid_loss.append(val_losses.avg)
        train_loss.append(train_losses.avg)

        if early_stopping(train_loss, patience=10, min_delta=1.0):
            print('early stopping at %d epoch' % (epoch + 1))
            break

    print('------------------z model testing----------------------------')

    if os.path.isfile(model_dir + 'z_model_best.pth.tar'):
        checkpoint = torch.load(model_dir + 'z_model_best.pth.tar')
    else:
        raise ValueError('could not find checkpoint, make sure to train before test')

    model_z.load_state_dict(checkpoint['state_dict'])
    minimum_valid_loss = checkpoint['best_score']

    valid_losses, valid_losses_mae = get_score_2D(validloader, model_z, criterion, criterion_mae)
    valid_h_losses, valid_h_losses_mae = get_score_2D(validloader_h, model_z, criterion, criterion_mae)
    save_forecast_result(model_z, validset,validloader, log_dir, num_tracks=args.num_tracks, name='results_z_valid')
    save_forecast_result(model_z, testset,testloader, log_dir, num_tracks=args.num_tracks, name='results_z_test')
    save_log_2D(module_name, minimum_valid_loss, valid_losses.avg, valid_losses_mae.avg, valid_h_losses.avg,
             valid_h_losses_mae.avg, baseline_losses.avg, baseline_losses_mae.avg, baseline_losses_h.avg, baseline_losses_mae_h.avg,
             fig_name + '_z_alone', log_dir, args)




if __name__ == '__main__':

    if os.path.isdir(log_dir):
        pass
    else:
        os.mkdir(log_dir)

    args = launch_parser()

    print('loading dataset...')
    trainset, validset, testset = load_datasets(sample=1.0, list_params=('u', 'v', 'z'), load_t=load_t, valid_split=0.2,
                                                test_split=0.2, randomseed=47)
    trainset, validset, testset = dataset_filter_3D(trainset, validset, testset, args.hours, with_tracks=True,
                                                    num_tracks=args.num_tracks, normalize=True, levels=levels,
                                                    params=params)
    print('extracting hurricanes form valid...')
    validset_h = extract_hurricanes(validset)
    print('extracting hurricanes form test...')
    testset_h = extract_hurricanes(testset)

    print('getting baseline set...')
    valid_baseline_set = get_baselineset(validset)
    valid_baseline_set_h = get_baselineset(validset_h)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, sampler=None, batch_sampler=None,
                             num_workers=6)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
                             num_workers=6)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
                            num_workers=6)
    validloader_h = DataLoader(validset_h, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
                              num_workers=6)
    validloader_baseline = DataLoader(valid_baseline_set, batch_size=args.batch_size, shuffle=False, sampler=None,
                                     batch_sampler=None, num_workers=6)
    validloader_baseline_h = DataLoader(valid_baseline_set_h, batch_size=args.batch_size, shuffle=False, sampler=None,
                                       batch_sampler=None, num_workers=6)

    print('Performing 3 different trainings (checking the robustness of the model)')
    for i in range(3):
        if os.path.isdir(log_dir+str(i)+'/'):
            pass
        else:
            os.mkdir(log_dir+str(i)+'/')
        main(log_dir=log_dir+str(i)+'/', validset=validset, testset=testset,
             trainloader=trainloader, validloader=validloader, testloader=testloader, validloader_h=validloader_h,
             validloader_baseline=validloader_baseline, validloader_baseline_h=validloader_baseline_h, args=args)
