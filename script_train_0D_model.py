import sys
#sys.path.append('/home/tau/sgiffard/ClimateMo/ClimateSaclayRepo/')
import os
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import Model_0D as baseModels

from Utils.loss import Regress_Loss, Regress_Loss_Mae
from Utils.Utils import save_checkpoint, dataset_filter_3D, extract_hurricanes, get_baselineset
from Utils.funcs import *
from Utils.model_inputs_3D import load_datasets
from Utils.save_figs_logs import save_log_0D, save_fig_0D, save_forecast_result_0D

######################
# hyper-parameters to set here
######################

data_dir = "/data/titanic_1/users/sophia/myang/model/data_3d_uvz_historic12h/"
log_dir = "/data/titanic_1/users/sophia/sgiffard/Tracking_res/CNN_res/logs/0D_24h_allmeta_standard/"

levels = (2, 5)
params = []
uv_params = ()
z_params = ()
load_t = [0]
meta = True  # add other meta data

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
                        help='predition hours: ' + ' (default: 24)')
        pr.add_argument('--num_tracks', default=2, type=int, metavar='NT',
                        choices=num_tracks,
                        help='number of past track times in y label (default: 2)')
        pr.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
        pr.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
        pr.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
        pr.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
        pr.add_argument('--betas', default=(0.9, 0.999), type=float, metavar='B',
                        help='betas')
        pr.add_argument('--eps', default=1e-8, type=float, metavar='EPS',
                        help='eps')
        pr.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
        pr.add_argument('--save_fig_name', default='0', type=str, metavar='N',
                        help='give the name of saved fig and log')

    arguments = parser.parse_args()
    return arguments

##############################
# main function (training the model)
##############################


def main(log_dir, validset, testset, trainloader, validloader, testloader, validloader_baseline, args):
    best_score = 1e10
    fig_name = args.save_fig_name
    if os.path.isdir(log_dir+'models'+'/'):
        pass
    else:
        os.mkdir(log_dir+'models'+'/')
    model_dir = log_dir+'models'+'/'

    criterion = Regress_Loss()
    criterion_mae = Regress_Loss_Mae()

    print('--------------------------0D model training num_tracks=2, with_meta=False---------------------------')
    model_0_d = baseModels.Base_model_disp_onlytracks(dropout=0.0, num_tracks=2, with_windspeed=meta).double().cuda()
    model_0_d = nn.DataParallel(model_0_d).cuda()
    optimizer = optim.Adam(model_0_d.parameters(), lr=args.lr, betas=args.betas, eps=args.eps,
                           weight_decay=args.weight_decay)
    train_loss = []
    valid_loss = []
    test_loss = []
    for epoch in range(0, args.epochs):
        print('start training with epoch(%d/%5d)' % (epoch + 1, args.epochs))

        # train for one epoch
        train_0D(trainloader, model_0_d, criterion, optimizer, epoch, meta=meta)

        # validate for the epoch
        train_losses, train_losses_mae = get_score_0D(trainloader, model_0_d, criterion, criterion_mae, meta=meta)
        val_losses, val_losses_mae = get_score_0D(validloader, model_0_d, criterion, criterion_mae, meta=meta)
        test_losses, test_losses_mae = get_score_0D(testloader, model_0_d, criterion, criterion_mae, meta=meta)

        print('valid loss:', val_losses.avg)
        print('test loss:', test_losses.avg)
        score = val_losses.avg
        is_best = score < best_score  # smaller the better
        best_score = min(score, best_score)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_0_d.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=model_dir, name='0D_model_24h_2t_best.pth.tar',
            filename_temp='sophie_checkpoint.pth.tar')
        valid_loss.append(val_losses.avg)
        train_loss.append(train_losses.avg)
        test_loss.append(test_losses.avg)

    if os.path.isfile(model_dir + '0D_model_24h_2t_best.pth.tar'):
        checkpoint = torch.load(model_dir + '0D_model_24h_2t_best.pth.tar')
    else:
        raise ValueError('could not find checkpoint')
    save_fig_0D(model_0_d.__class__.__name__, train_loss, valid_loss, fig_name, log_dir, args)

    print('------------------0D model testing----------------------------')
    model_0_d.load_state_dict(checkpoint['state_dict'])
    minimum_valid_loss = checkpoint['best_score']

    print('extracting hurricanes form valid...')
    validset_h = extract_hurricanes(validset)
    print('extracting hurricanes form test...')
    testset_h = extract_hurricanes(testset)
    valid_baseline_set_h = get_baselineset(validset_h)

    validloader_baseline_h = DataLoader(valid_baseline_set_h, batch_size=args.batch_size, shuffle=False, sampler=None,
                                       batch_sampler=None, num_workers=6)
    testloader_h = DataLoader(testset_h, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
                              num_workers=6)
    validloader_h = DataLoader(validset_h, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
                              num_workers=6)

    baseline_losses, baseline_losses_mae = get_baseline_score(validloader_baseline, criterion, criterion_mae,
                                                              hours=args.hours)
    baseline_losses_h, baseline_losses_mae_h = get_baseline_score(validloader_baseline_h, criterion, criterion_mae,
                                                                  hours=args.hours)
    valid_losses, valid_losses_mae = get_score_0D(validloader, model_0_d, criterion, criterion_mae, meta=meta)
    valid_h_losses, valid_h_losses_mae = get_score_0D(validloader_h, model_0_d, criterion, criterion_mae, meta=meta)

    test_losses, test_losses_mae = get_score_0D(testloader, model_0_d, criterion, criterion_mae, meta=meta)
    test_h_losses, test_h_losses_mae = get_score_0D(testloader_h, model_0_d, criterion, criterion_mae, meta=meta)

    save_log_0D(model_0_d.__class__.__name__, minimum_valid_loss, valid_losses.avg, valid_losses_mae.avg, valid_h_losses.avg,
             valid_h_losses_mae.avg,
             baseline_losses.avg, baseline_losses_mae.avg, baseline_losses_h.avg, baseline_losses_mae_h.avg,
             fig_name, log_dir, test_losses.avg, test_losses_mae.avg, test_h_losses.avg,
             test_h_losses_mae.avg, args)

    save_forecast_result_0D(model_0_d, validset, validloader, log_dir, num_tracks=2, name='results_0D_24h_2t_valid')
    save_forecast_result_0D(model_0_d, testset, testloader, log_dir, num_tracks=2, name='results_0D_24h_2t_test' )


if __name__ == '__main__':

    if os.path.isdir(log_dir):
        pass
    else:
        os.mkdir(log_dir)

    args = launch_parser()

    print('loading dataset...')
    trainset, validset, testset = load_datasets(sample=1.0, list_params=('z'), load_t=load_t, valid_split=0.2, test_split=0.2, randomseed=47)
    trainset, validset, testset = dataset_filter_3D(trainset, validset, testset, args.hours, with_tracks=True,
                                                    num_tracks=2, normalize=True, levels=levels,
                                                    params=params, with_windspeed=True)
    print('getting baseline set...')
    test_baseline_set = get_baselineset(testset)
    valid_baseline_set = get_baselineset(validset)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, sampler=None, batch_sampler=None,
                             num_workers=6)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
                             num_workers=6)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
                            num_workers=6)
    validloader_baseline = DataLoader(valid_baseline_set, batch_size=args.batch_size, shuffle=False, sampler=None,
                                     batch_sampler=None, num_workers=6)

    print('Performing 3 different trainings (checking the robustness of the model)')
    for i in range(3):
        if os.path.isdir(log_dir+str(i)+'/'):
            pass
        else:
            os.mkdir(log_dir+str(i)+'/')
        main(log_dir=log_dir+str(i)+'/',  validset=validset, testset=testset,
             trainloader=trainloader, validloader=validloader, testloader=testloader,
             validloader_baseline=validloader_baseline, args=args)
