import sys
#sys.path.append('/home/tau/myang/ClimateSaclayRepo/')
import os
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import Model_uv
import Model_z
import Model_fuseall
import Model_0D

from Utils.loss import Regress_Loss, Regress_Loss_Mae
from Utils.Utils import save_checkpoint, early_stopping, dataset_filter_3D
from Utils.funcs import *
from Utils.model_inputs_3D import load_datasets

##############################
# paths and hard-coded parameters
##############################

data_dir = "/data/titanic_1/users/sophia/myang/model/data_3d_uvz_historic12h/"
log_dir = "/data/titanic_1/users/sophia/myang/logs/3conv_models_24h/"
log_dir_fusion = log_dir + 'fusion_3conv_uvz0d_fuse3fc_freeze_200/'
model_dir = log_dir+'models'+'/'

# paths to the 3 saved independent models trained (.tar)
uv_model_tar = model_dir  + 'uv_model_best.pth.tar'
z_model_tar = model_dir + 'z_model_best.pth.tar'
_0D_model_tar = model_dir  + '0D_model_24h_2t_allmeta_best.pth.tar'

# path for saving the fused model (.tar)
name_tar_fused_freeze = 'fusion_3fc_uvz0d_model_best_freeze.pth.tar'

levels = (2, 5, 7)
params = (0,1,2,3,4,5,6,7,8)
uv_params = (0,1,3,4,6,7)
z_params = (2,5,8)
# params: 0:u 1:v, 2:z
load_t = (0,-6,-12)
arch = "Net_2d_conv3"
meta = True

##############################
# Arguments
##############################

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
    for pr in parsers:
        pr.add_argument('--hours', metavar='HOURS', default=24,
                        type=int, choices=predict_hours,
                        help='predition hours: (default: 24)')
        pr.add_argument('--epochs_freeze', default=90, type=int, metavar='N',
                        help='number of total epochs for the freezing part')
        pr.add_argument('--epochs_final', default=90, type=int, metavar='N',
                        help='number of total epochs for the final part')
        pr.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
        pr.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
        pr.add_argument('--betas', default=(0.9, 0.999), type=float, metavar='B',
                        help='betas')
        pr.add_argument('--eps', default=1e-8, type=float, metavar='EPS',
                        help='eps')
        pr.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    arguments = parser.parse_args()
    return arguments


def main(trainloader, validloader, testloader, args):

    criterion = Regress_Loss()
    criterion_mae = Regress_Loss_Mae()

    print('------------------------load_uv_model---------------------------')
    model_uv = Model_uv.__dict__[arch](dropout=0.0, levellist=levels, params=uv_params).double().cuda()
    model_uv = nn.DataParallel(model_uv).cuda()
    if os.path.isfile(uv_model_tar):
        checkpoint = torch.load(uv_model_tar)
    else:
        raise ValueError('could not find checkpoint')
    model_uv.load_state_dict(checkpoint['state_dict'])

    print('------------------------load_z_model---------------------------')
    model_z = Model_z.__dict__[arch](dropout=0.0, levellist=levels, params=z_params).double().cuda()
    model_z = nn.DataParallel(model_z).cuda()
    if os.path.isfile(z_model_tar):
        checkpoint = torch.load(z_model_tar)
    else:
        raise ValueError('could not find checkpoint, make sure to train before test')
    model_z.load_state_dict(checkpoint['state_dict'])

    print('-------------------------load_0d_model--------------------------')
    model_0_d = Model_0D.Base_model_disp_onlytracks(dropout=0.0, num_tracks=2, with_windspeed=meta).double().cuda()
    model_0_d = nn.DataParallel(model_0_d).cuda()
    if os.path.isfile(_0D_model_tar):
        checkpoint = torch.load(_0D_model_tar)
    else:
        raise ValueError('could not find checkpoint')
    model_0_d.load_state_dict(checkpoint['state_dict'])

    print('------------------------train uvz fusion model fuse conv layer---------------------------')
    model_fusion = Model_fuseall.Net_2d_conv3_fuse_3fc(dropout=0.0, levellist=levels, params=params).double().cuda()
    model_fusion = nn.DataParallel(model_fusion).cuda()
    pretrained_dict_uv = model_uv.state_dict()
    pretrained_dict_z = model_z.state_dict()
    pretrained_dict_0d = model_0_d.state_dict()
    model_dict = model_fusion.state_dict()

    # LOAD THE WEIGHTS ALREADY TRAINED #
    # load weigths for uv
    model_dict['module.conv1_uv.weight'] = pretrained_dict_uv['module.conv1.weight']
    model_dict['module.conv1_uv.bias'] = pretrained_dict_uv['module.conv1.bias']
    model_dict['module.conv1_bn_uv.weight'] = pretrained_dict_uv['module.conv1_bn.weight']
    model_dict['module.conv1_bn_uv.bias'] = pretrained_dict_uv['module.conv1_bn.bias']
    model_dict['module.conv1_bn_uv.running_mean'] = pretrained_dict_uv['module.conv1_bn.running_mean']
    model_dict['module.conv1_bn_uv.running_var'] = pretrained_dict_uv['module.conv1_bn.running_var']

    model_dict['module.conv2_uv.weight'] = pretrained_dict_uv['module.conv2.weight']
    model_dict['module.conv2_uv.bias'] = pretrained_dict_uv['module.conv2.bias']
    model_dict['module.conv2_bn_uv.weight'] = pretrained_dict_uv['module.conv2_bn.weight']
    model_dict['module.conv2_bn_uv.bias'] = pretrained_dict_uv['module.conv2_bn.bias']
    model_dict['module.conv2_bn_uv.running_mean'] = pretrained_dict_uv['module.conv2_bn.running_mean']
    model_dict['module.conv2_bn_uv.running_var'] = pretrained_dict_uv['module.conv2_bn.running_var']

    model_dict['module.conv3_uv.weight'] = pretrained_dict_uv['module.conv3.weight']
    model_dict['module.conv3_uv.bias'] = pretrained_dict_uv['module.conv3.bias']
    model_dict['module.conv3_bn_uv.weight'] = pretrained_dict_uv['module.conv3_bn.weight']
    model_dict['module.conv3_bn_uv.bias'] = pretrained_dict_uv['module.conv3_bn.bias']
    model_dict['module.conv3_bn_uv.running_mean'] = pretrained_dict_uv['module.conv3_bn.running_mean']
    model_dict['module.conv3_bn_uv.running_var'] = pretrained_dict_uv['module.conv3_bn.running_var']

    # load weights for z
    model_dict['module.conv1_z.weight'] = pretrained_dict_z['module.conv1.weight']
    model_dict['module.conv1_z.bias'] = pretrained_dict_z['module.conv1.bias']
    model_dict['module.conv1_bn_z.weight'] = pretrained_dict_z['module.conv1_bn.weight']
    model_dict['module.conv1_bn_z.bias'] = pretrained_dict_z['module.conv1_bn.bias']
    model_dict['module.conv1_bn_z.running_mean'] = pretrained_dict_z['module.conv1_bn.running_mean']
    model_dict['module.conv1_bn_z.running_var'] = pretrained_dict_z['module.conv1_bn.running_var']

    model_dict['module.conv2_z.weight'] = pretrained_dict_z['module.conv2.weight']
    model_dict['module.conv2_z.bias'] = pretrained_dict_z['module.conv2.bias']
    model_dict['module.conv2_bn_z.weight'] = pretrained_dict_z['module.conv2_bn.weight']
    model_dict['module.conv2_bn_z.bias'] = pretrained_dict_z['module.conv2_bn.bias']
    model_dict['module.conv2_bn_z.running_mean'] = pretrained_dict_z['module.conv2_bn.running_mean']
    model_dict['module.conv2_bn_z.running_var'] = pretrained_dict_z['module.conv2_bn.running_var']

    model_dict['module.conv3_z.weight'] = pretrained_dict_z['module.conv3.weight']
    model_dict['module.conv3_z.bias'] = pretrained_dict_z['module.conv3.bias']
    model_dict['module.conv3_bn_z.weight'] = pretrained_dict_z['module.conv3_bn.weight']
    model_dict['module.conv3_bn_z.bias'] = pretrained_dict_z['module.conv3_bn.bias']
    model_dict['module.conv3_bn_z.running_mean'] = pretrained_dict_z['module.conv3_bn.running_mean']
    model_dict['module.conv3_bn_z.running_var'] = pretrained_dict_z['module.conv3_bn.running_var']

    # fc layers uv
    model_dict['module.fc1_uv.weight'] = pretrained_dict_uv['module.fc1.weight']
    model_dict['module.fc1_uv.bias'] = pretrained_dict_uv['module.fc1.bias']
    model_dict['module.fc1_bn_uv.weight'] = pretrained_dict_uv['module.fc1_bn.weight']
    model_dict['module.fc1_bn_uv.bias'] = pretrained_dict_uv['module.fc1_bn.bias']
    model_dict['module.fc1_bn_uv.running_mean'] = pretrained_dict_uv['module.fc1_bn.running_mean']
    model_dict['module.fc1_bn_uv.running_var'] = pretrained_dict_uv['module.fc1_bn.running_var']

    model_dict['module.fc2_uv.weight'] = pretrained_dict_uv['module.fc2.weight']
    model_dict['module.fc2_uv.bias'] = pretrained_dict_uv['module.fc2.bias']
    model_dict['module.fc2_bn_uv.weight'] = pretrained_dict_uv['module.fc2_bn.weight']
    model_dict['module.fc2_bn_uv.bias'] = pretrained_dict_uv['module.fc2_bn.bias']
    model_dict['module.fc2_bn_uv.running_mean'] = pretrained_dict_uv['module.fc2_bn.running_mean']
    model_dict['module.fc2_bn_uv.running_var'] = pretrained_dict_uv['module.fc2_bn.running_var']

    model_dict['module.fc3_uv.weight'] = pretrained_dict_uv['module.fc3.weight']
    model_dict['module.fc3_uv.bias'] = pretrained_dict_uv['module.fc3.bias']
    model_dict['module.fc3_bn_uv.weight'] = pretrained_dict_uv['module.fc3_bn.weight']
    model_dict['module.fc3_bn_uv.bias'] = pretrained_dict_uv['module.fc3_bn.bias']
    model_dict['module.fc3_bn_uv.running_mean'] = pretrained_dict_uv['module.fc3_bn.running_mean']
    model_dict['module.fc3_bn_uv.running_var'] = pretrained_dict_uv['module.fc3_bn.running_var']

    model_dict['module.fc4_uv.weight'] = pretrained_dict_uv['module.fc4.weight']
    model_dict['module.fc4_uv.bias'] = pretrained_dict_uv['module.fc4.bias']
    model_dict['module.fc4_bn_uv.weight'] = pretrained_dict_uv['module.fc4_bn.weight']
    model_dict['module.fc4_bn_uv.bias'] = pretrained_dict_uv['module.fc4_bn.bias']
    model_dict['module.fc4_bn_uv.running_mean'] = pretrained_dict_uv['module.fc4_bn.running_mean']
    model_dict['module.fc4_bn_uv.running_var'] = pretrained_dict_uv['module.fc4_bn.running_var']

    # fc layers z
    model_dict['module.fc1_z.weight'] = pretrained_dict_z['module.fc1.weight']
    model_dict['module.fc1_z.bias'] = pretrained_dict_z['module.fc1.bias']
    model_dict['module.fc1_bn_z.weight'] = pretrained_dict_z['module.fc1_bn.weight']
    model_dict['module.fc1_bn_z.bias'] = pretrained_dict_z['module.fc1_bn.bias']
    model_dict['module.fc1_bn_z.running_mean'] = pretrained_dict_z['module.fc1_bn.running_mean']
    model_dict['module.fc1_bn_z.running_var'] = pretrained_dict_z['module.fc1_bn.running_var']

    model_dict['module.fc2_z.weight'] = pretrained_dict_z['module.fc2.weight']
    model_dict['module.fc2_z.bias'] = pretrained_dict_z['module.fc2.bias']
    model_dict['module.fc2_bn_z.weight'] = pretrained_dict_z['module.fc2_bn.weight']
    model_dict['module.fc2_bn_z.bias'] = pretrained_dict_z['module.fc2_bn.bias']
    model_dict['module.fc2_bn_z.running_mean'] = pretrained_dict_z['module.fc2_bn.running_mean']
    model_dict['module.fc2_bn_z.running_var'] = pretrained_dict_z['module.fc2_bn.running_var']

    model_dict['module.fc3_z.weight'] = pretrained_dict_z['module.fc3.weight']
    model_dict['module.fc3_z.bias'] = pretrained_dict_z['module.fc3.bias']
    model_dict['module.fc3_bn_z.weight'] = pretrained_dict_z['module.fc3_bn.weight']
    model_dict['module.fc3_bn_z.bias'] = pretrained_dict_z['module.fc3_bn.bias']
    model_dict['module.fc3_bn_z.running_mean'] = pretrained_dict_z['module.fc3_bn.running_mean']
    model_dict['module.fc3_bn_z.running_var'] = pretrained_dict_z['module.fc3_bn.running_var']

    model_dict['module.fc4_z.weight'] = pretrained_dict_z['module.fc4.weight']
    model_dict['module.fc4_z.bias'] = pretrained_dict_z['module.fc4.bias']
    model_dict['module.fc4_bn_z.weight'] = pretrained_dict_z['module.fc4_bn.weight']
    model_dict['module.fc4_bn_z.bias'] = pretrained_dict_z['module.fc4_bn.bias']
    model_dict['module.fc4_bn_z.running_mean'] = pretrained_dict_z['module.fc4_bn.running_mean']
    model_dict['module.fc4_bn_z.running_var'] = pretrained_dict_z['module.fc4_bn.running_var']

    #  FOR THE FUSED LAYERS: CONCATNATE TRAINED LAYERS AND ADD 0 WEIGHTS FOR CROSS-STREAM CONNECTIONS ###
    param_fc5_weights = torch.zeros([8+8+8, 8+8+9]).cuda().double()
    param_fc5_weights[:8, :8] = pretrained_dict_uv['module.fc5.weight']
    param_fc5_weights[8:16, 8:16] = pretrained_dict_z['module.fc5.weight']
    param_fc5_weights[16:, 16:] = pretrained_dict_0d['module.fc1.weight']

    param_fc5_bias = torch.zeros([8*3]).cuda().double()
    param_fc5_bias[:8] = pretrained_dict_uv['module.fc5.bias']
    param_fc5_bias[8:16] = pretrained_dict_z['module.fc5.bias']
    param_fc5_bias[16:] = pretrained_dict_0d['module.fc1.bias']

    model_dict['module.fc5.weight'] = param_fc5_weights
    model_dict['module.fc5.bias'] = param_fc5_bias

    param_fc6_weights = torch.zeros([2*3, 8*3]).cuda().double()
    param_fc6_weights[:2, :8] = pretrained_dict_uv['module.fc6.weight']
    param_fc6_weights[2:4, 8:16] = pretrained_dict_z['module.fc6.weight']
    param_fc6_weights[4:,16:] = pretrained_dict_0d['module.fc2.weight']

    param_fc6_bias = torch.zeros([2*3]).cuda().double()
    param_fc6_bias[:2] = pretrained_dict_uv['module.fc6.bias']
    param_fc6_bias[2:4] = pretrained_dict_z['module.fc6.bias']
    param_fc6_bias[4:] = pretrained_dict_0d['module.fc2.bias']

    model_dict['module.fc6.weight'] = param_fc6_weights
    model_dict['module.fc6.bias'] = param_fc6_bias

    # ADD ONE LAYER AT THE END
    param_fc7_weights = torch.Tensor([[1/3, 0, 1/3, 0, 1/3, 0], [0, 1/3, 0, 1/3, 0, 1/3]]).cuda().double()
    param_fc7_bias = torch.zeros([2]).cuda().double()
    model_dict['module.fc7.weight'] = param_fc7_weights
    model_dict['module.fc7.bias'] = param_fc7_bias

    # load weigths for fusion model
    model_fusion.load_state_dict(model_dict)
    # set unfused layers freezed
    num_params = 0
    for param in model_fusion.parameters():
        num_params += 1
    num_unfreezed_params = len(('module.fc5.weight', 'module.fc6.weight', 'module.fc7.weight', 'module.fc5.bias',
                                'module.fc6.bias', 'module.fc7.bias'))
    for counter, param in enumerate(model_fusion.parameters()):
        if counter >= num_params-num_unfreezed_params:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_fusion.parameters()), lr=args.lr,
                           betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)

    train_loss = []
    valid_loss = []

    print('------------------start fusion model training while freezing the former layers----------------------------')
    best_score = 1e10
    for epoch in range(0, args.epochs_freeze):
        train_fusion(trainloader, model_fusion, criterion, optimizer, epoch, meta=meta)
        train_losses, train_losses_mae = get_score_fusion(trainloader, model_fusion, criterion, criterion_mae, meta=meta)
        val_losses, val_losses_mae = get_score_fusion(validloader, model_fusion, criterion, criterion_mae, meta=meta)

        print('valid loss:', val_losses.avg)
        score = val_losses.avg
        is_best = score < best_score  # smaller the better
        best_score = min(score, best_score)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_fusion.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=model_dir, name=name_tar_fused_freeze)
        valid_loss.append(val_losses.avg)
        train_loss.append(train_losses.avg)

        if early_stopping(train_loss, patience=10, min_delta=1.0):
            print('early stopping at %d epoch' % (epoch + 1))
            break

    print('------------------continue training without freezing----------------------------')
    print('start testing 1/1')
    if os.path.isfile(model_dir + name_tar_fused_freeze):
        checkpoint = torch.load(model_dir + name_tar_fused_freeze)
    else:
        raise ValueError('could not find checkpoint, make sure to train before test')
    model_fusion.load_state_dict(checkpoint['state_dict'])
    for param in model_fusion.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model_fusion.parameters(), lr=args.lr, betas=args.betas, eps=args.eps,
                           weight_decay=args.weight_decay)
    for epoch in range(0, args.epochs_final):
        train_fusion(trainloader, model_fusion, criterion, optimizer, epoch, meta=meta)
        train_losses, train_losses_mae = get_score_fusion(trainloader, model_fusion, criterion, criterion_mae, meta=meta)
        val_losses, val_losses_mae = get_score_fusion(validloader, model_fusion, criterion, criterion_mae, meta=meta)

        print('valid loss:', val_losses.avg)
        score = val_losses.avg
        is_best = score < best_score  # smaller the better
        best_score = min(score, best_score)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_fusion.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=model_dir, name=name_tar_fused_freeze)
        valid_loss.append(val_losses.avg)
        train_loss.append(train_losses.avg)

        if early_stopping(train_loss, patience=10, min_delta=1.0):
            print('early stopping at %d epoch' % (epoch + 1))
            break

    print('------------------fusion testing----------------------------')
    print('start testing 1/1')
    if os.path.isfile(model_dir + name_tar_fused_freeze):
        checkpoint = torch.load(model_dir + name_tar_fused_freeze)
    else:
        raise ValueError('could not find checkpoint, make sure to train before test')
    model_fusion.load_state_dict(checkpoint['state_dict'])
    minimum_valid_loss = checkpoint['best_score']

    valid_losses, valid_losses_mae = get_score_fusion(validloader, model_fusion, criterion, criterion_mae, meta=meta)

    test_losses, test_losses_mae = get_score_fusion(testloader, model_fusion, criterion, criterion_mae, meta=meta)

    print('------------------finished testing----------------------------')

    print('save your results here!')


if __name__ == '__main__':

    args = launch_parser()

    print('loading dataset...')
    trainset, validset, testset = load_datasets(sample=1.0, list_params=('u', 'v', 'z'),
                                                load_t=load_t, valid_split=0.2, test_split=0.2, randomseed=47)
    trainset, validset, testset = dataset_filter_3D(trainset, validset, testset, args.hours, with_tracks=True,
                                                    num_tracks=2, normalize=True, levels=levels,
                                                    params=params, with_windspeed=True)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, sampler=None, batch_sampler=None,
                             num_workers=6)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
                             num_workers=6)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
                            num_workers=6)

    main(trainloader=trainloader, validloader=validloader, testloader=testloader, args=args)
