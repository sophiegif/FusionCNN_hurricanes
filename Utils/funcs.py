from Utils.Utils import AverageMeter
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from Utils.MyDataset import MyDataset
from Utils import loss_notorch
import pandas as pd
import pickle


def train_0D(train_loader, model, criterion, optimizer, epoch, num_tracks=2, meta=False):
    '''

    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param num_tracks: number of tails
    :param meta: wether or not add meta data into tracks
    :return:
    '''

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        if meta == False:
            input = target[:,-num_tracks:,:].view(target.shape[0],-1)
        else:
            # add windspeed and J-day, long, lat (standardized)
            input = target[:,-(num_tracks+3):-1,:].view(target.shape[0],-1)
            # add dist2land
            dist2land = target[:,-1,0].view(target.shape[0],-1)
            input = torch.cat((input, dist2land), 1)
            #print('input: '+str(input[0]),flush=True)
            #print('target:' + str(target[0]) ,flush=True)
        # compute output and loss
        output = model(input)
        loss = criterion(output, target)
        # record loss
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d/%d] train loss: %.5f '%
                (epoch + 1, i + 1, len(train_loader), losses.avg))
        if i==len(train_loader)-1:
            print('[%d, %5d/%d] loss: %.5f'%
                (epoch+1, i+1, len(train_loader), losses.avg))
    print(  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.avg:.5f}\t'
            .format(batch_time=batch_time, data_time=data_time, loss=losses))


def train_2D(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        # compute output and loss
        output = model(input)
        loss = criterion(output, target)
        # record loss
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        #optimizer.module.step()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d/%d] train loss: %.5f '%
                (epoch + 1, i + 1, len(train_loader), losses.avg))
        if i==len(train_loader)-1:
            print('[%d, %5d/%d] loss: %.5f'%
                (epoch+1, i+1, len(train_loader), losses.avg))
    print(  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.avg:.5f}\t'
            .format(batch_time=batch_time, data_time=data_time, loss=losses))


def train_2D_uv_z(trainloader_uv, trainloader_z, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, ((input_uv, target), (input_z, _)) in enumerate(zip(trainloader_uv, trainloader_z)):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        # compute output and loss
        output = model(input_uv, input_z)
        loss = criterion(output, target)
        # record loss
        losses.update(loss.item(), target.size(0))
        optimizer.zero_grad()
        loss.backward()
        #optimizer.module.step()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d/%d] train loss: %.5f '%
                (epoch + 1, i + 1, len(trainloader_uv), losses.avg))
        if i==len(trainloader_uv)-1:
            print('[%d, %5d/%d] loss: %.5f'%
                (epoch+1, i+1, len(trainloader_uv), losses.avg))
    print(  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.avg:.5f}\t'
            .format(batch_time=batch_time, data_time=data_time, loss=losses))


def train_fusion(train_loader, model, criterion, optimizer, epoch,num_tracks=2, meta=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        # compute output and loss
        if meta == False:
            input_0d = target[:,-num_tracks:,:].view(target.shape[0],-1)
        else:
            input_0d = target[:,-(num_tracks+3):-1,:].view(target.shape[0],-1)
            # add dist2land
            dist2land = target[:, -1, 0].view(target.shape[0], -1)
            input_0d = torch.cat((input_0d, dist2land), 1)
        output = model(input, input_0d)
        loss = criterion(output, target)
        # record loss
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d/%d] train loss: %.5f '%
                (epoch + 1, i + 1, len(train_loader), losses.avg))
        if i==len(train_loader)-1:
            print('[%d, %5d/%d] loss: %.5f'%
                (epoch+1, i+1, len(train_loader), losses.avg))
    print(  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.avg:.5f}\t'
            .format(batch_time=batch_time, data_time=data_time, loss=losses))


def get_baseline_score(test_loader_baseline, criterion, criterion_mae, hours=6):
    losses = AverageMeter()
    losses_mae = AverageMeter()
    mul_factor = hours/6
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader_baseline):
            output = input*mul_factor
            #target = target.cuda(non_blocking=True)
            loss = criterion(output, target)
            loss_mae = criterion_mae(output, target)
            losses.update(loss.item(), input.size(0))
            losses_mae.update(loss_mae.item(), input.size(0))
    return losses, losses_mae


def get_baseline_score_(valid_baseline_set, criterion, criterion_mae, hours=6):
    losses = 0
    losses_mae = 0
    mul_factor = hours/6
    counter = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_baseline_set):
            output = input*mul_factor
            #target = target.cuda(non_blocking=True)
            losses += criterion(output, target)
            losses_mae += criterion_mae(output, target)
            counter += 1

    return losses/counter, losses_mae/counter


def get_score_0D(val_loader, model, criterion, criterion_mae,num_tracks=2, multio = False, meta = False):
    if multio == False:
        losses = AverageMeter()
        losses_mae = AverageMeter()
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                # compute output
                if meta == False:
                    input = target[:,-num_tracks:,:].view(target.shape[0],-1)
                else:
                    # add windspeed and J-day and long lat
                    input = target[:, -(num_tracks + 3):-1, :].view(target.shape[0], -1)
                    # add dist2land
                    dist2land = target[:, -1, 0].view(target.shape[0], -1)
                    input = torch.cat((input, dist2land), 1)

                output = model(input)
                loss = criterion(output, target)
                loss_mae = criterion_mae(output, target)
                # record loss
                losses.update(loss.item(), input.size(0))
                losses_mae.update(loss_mae.item(), input.size(0))
        return losses, losses_mae
    else:
        losses = AverageMeter()
        losses_mae_6h = AverageMeter()
        losses_mae_12h = AverageMeter()
        losses_mae_18h = AverageMeter()
        losses_mae_24h = AverageMeter()
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                # compute output
                input = target[:,-num_tracks:,:].view(target.shape[0],-1)
                output = model(input)
                loss = criterion(output, target)
                loss_mae_6h, loss_mae_12h, loss_mae_18h, loss_mae_24h = criterion_mae(output, target)
                # record loss
                losses.update(loss.item(), input.size(0))
                losses_mae_6h.update(loss_mae_6h.item(), input.size(0))
                losses_mae_12h.update(loss_mae_12h.item(), input.size(0))
                losses_mae_18h.update(loss_mae_18h.item(), input.size(0))
                losses_mae_24h.update(loss_mae_24h.item(), input.size(0))

        return losses, losses_mae_6h,losses_mae_12h,losses_mae_18h, losses_mae_24h


def get_score_2D(val_loader, model, criterion, criterion_mae, multio = False):
    if multio == False:
        losses = AverageMeter()
        losses_mae = AverageMeter()
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                # compute output
                output = model(input)
                loss = criterion(output, target)
                loss_mae = criterion_mae(output, target)
                # record loss
                losses.update(loss.item(), input.size(0))
                losses_mae.update(loss_mae.item(), input.size(0))
        return losses, losses_mae
    else:
        losses = AverageMeter()
        losses_mae_6h = AverageMeter()
        losses_mae_12h = AverageMeter()
        losses_mae_18h = AverageMeter()
        losses_mae_24h = AverageMeter()
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                # compute output
                output = model(input)
                loss = criterion(output, target)
                loss_mae_6h, loss_mae_12h, loss_mae_18h, loss_mae_24h = criterion_mae(output, target)
                # record loss
                losses.update(loss.item(), input.size(0))
                losses_mae_6h.update(loss_mae_6h.item(), input.size(0))
                losses_mae_12h.update(loss_mae_12h.item(), input.size(0))
                losses_mae_18h.update(loss_mae_18h.item(), input.size(0))
                losses_mae_24h.update(loss_mae_24h.item(), input.size(0))

        return losses, losses_mae_6h,losses_mae_12h,losses_mae_18h, losses_mae_24h


def get_score_2D_uv_z(validloader_uv, validloader_z, model, criterion, criterion_mae):
    losses = AverageMeter()
    losses_mae = AverageMeter()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, ((input_uv, target),(input_z, _)) in enumerate(zip(validloader_uv, validloader_z)):
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(input_uv, input_z)
            loss = criterion(output, target)
            loss_mae = criterion_mae(output, target)
            # record loss
            losses.update(loss.item(), target.size(0))
            losses_mae.update(loss_mae.item(), target.size(0))
    return losses, losses_mae



def get_score_fusion(val_loader, model, criterion, criterion_mae, num_tracks=2, multio = False, meta=False):
    if multio == False:
        losses = AverageMeter()
        losses_mae = AverageMeter()
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                # compute output
                if meta == False:
                    input_0d = target[:, -num_tracks:, :].view(target.shape[0], -1)
                else:
                    input_0d = target[:, -(num_tracks + 3):-1, :].view(target.shape[0], -1)
                    # add dist2land
                    dist2land = target[:, -1, 0].view(target.shape[0], -1)
                    input_0d = torch.cat((input_0d, dist2land), 1)
                output = model(input, input_0d)
                loss = criterion(output, target)
                loss_mae = criterion_mae(output, target)
                # record loss
                losses.update(loss.item(), input.size(0))
                losses_mae.update(loss_mae.item(), input.size(0))
        return losses, losses_mae
    else:
        losses = AverageMeter()
        losses_mae_6h = AverageMeter()
        losses_mae_12h = AverageMeter()
        losses_mae_18h = AverageMeter()
        losses_mae_24h = AverageMeter()
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                # compute output
                if meta == False:
                    input_0d = target[:, -num_tracks:, :].view(target.shape[0], -1)
                else:
                    input_0d = target[:, -(num_tracks + 3):-1, :].view(target.shape[0], -1)
                output = model(input, input_0d)
                loss = criterion(output, target)
                loss_mae_6h, loss_mae_12h, loss_mae_18h, loss_mae_24h = criterion_mae(output, target)
                # record loss
                losses.update(loss.item(), input.size(0))
                losses_mae_6h.update(loss_mae_6h.item(), input.size(0))
                losses_mae_12h.update(loss_mae_12h.item(), input.size(0))
                losses_mae_18h.update(loss_mae_18h.item(), input.size(0))
                losses_mae_24h.update(loss_mae_24h.item(), input.size(0))

        return losses, losses_mae_6h, losses_mae_12h, losses_mae_18h, losses_mae_24h


def get_baseline_losses(dataset, criterion, criterion_mae, hours=6):
    losses_baseline = []
    losses_baseline_mae = []
    mul_factor = hours/6
    for i in range(len(dataset)):
        output = torch.Tensor(dataset.images[i:i+1]).cuda().double() * mul_factor
        target = torch.Tensor(dataset.labels[i:i+1]).cuda().double()
        loss = criterion(output, target).cpu().numpy()
        losses_baseline.append(loss)
        loss_mae = criterion_mae(output, target).cpu().numpy()
        losses_baseline_mae.append(loss_mae)
    return np.array(losses_baseline), np.array(losses_baseline_mae)


def get_0D_model_losses(dataset, model, criterion, criterion_mae, num_tracks=2):
    losses_basemodel = []
    losses_basemodel_mae = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            target = torch.Tensor(dataset.labels[i:i+1]).cuda().double()
            input = target[:,-num_tracks:,:].view(target.shape[0],-1)
            output = model(input)
            loss = criterion(output, target).cpu().numpy()
            losses_basemodel.append(loss)
            loss_mae = criterion_mae(output, target).cpu().numpy()
            losses_basemodel_mae.append(loss_mae)
    return np.array(losses_basemodel), np.array(losses_basemodel_mae)


def get_2D_model_losses(dataset, model, criterion, criterion_mae):
    losses = []
    losses_mae = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            input = torch.Tensor(dataset.images[i:i+1]).cuda().double()
            target = torch.Tensor(dataset.labels[i:i+1]).cuda().double()
            output = model(input)
            loss = criterion(output, target).cpu().numpy()
            losses.append(loss)
            loss_mae = criterion_mae(output, target).cpu().numpy()
            losses_mae.append(loss_mae)
    return np.array(losses), np.array(losses_mae)


def get_fusion_model_losses(dataset, model, criterion, criterion_mae,num_tracks=2):
    losses = []
    losses_mae = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            input = torch.Tensor(dataset.images[i:i+1]).cuda().double()
            target = torch.Tensor(dataset.labels[i:i+1]).cuda().double()
            output = model(input, target[:,-num_tracks:,:].view(target.shape[0],-1))
            loss = criterion(output, target).cpu().numpy()
            losses.append(loss)
            loss_mae = criterion_mae(output, target).cpu().numpy()
            losses_mae.append(loss_mae)
    return np.array(losses), np.array(losses_mae)


def save_boxplot_MSE(losses_baseline,losses_0D,losses_2D,losses_fusion, set_belong='whole_set', log_dir='/data/titanic_1/users/sophia/myang/logs/'):
    numModels = 4
    data = [losses_baseline,losses_2D,losses_0D,losses_fusion]

    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    bp = ax1.boxplot(data, notch=0, sym='', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='blue')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.setp(bp['medians'], color = 'black')
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of MSE from different models'+'(%s)' % set_belong,fontweight="bold", size=20)
    #ax1.set_xlabel('Models')
    ax1.set_ylabel('Squared error', size=15)
    ax1.tick_params(axis = 'y', which = 'major', labelsize = 13)
    # Now fill the boxes with desired colors
    box = bp['boxes'][0]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor='lightgrey')
    ax1.add_patch(boxPolygon)

    box = bp['boxes'][1]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor='lightgreen')
    ax1.add_patch(boxPolygon)

    box = bp['boxes'][2]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor='lightblue')
    ax1.add_patch(boxPolygon)

    box = bp['boxes'][3]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor='lightcoral')
    ax1.add_patch(boxPolygon)

    ax1.set_xticklabels(['BASELINE','ONLY WIND FIELD','ONLY PAST TRACKS','FUSION NETWORK'], size='12')


    fig.text(0.19, 0.8, ' MSE = %.1f' %(np.mean(losses_baseline)),
             backgroundcolor='whitesmoke', color='black', weight='roman',
             size='12')

    fig.text(0.38, 0.8, ' MSE = %.1f' %(np.mean(losses_2D)),
             backgroundcolor='whitesmoke', color='black', weight='roman',
             size='12')
    fig.text(0.58, 0.8, ' MSE = %.1f' %(np.mean(losses_0D)),
             backgroundcolor='whitesmoke', color='black', weight='roman',
             size='12')

    fig.text(0.78, 0.8, ' MSE = %.1f' %(np.mean(losses_fusion)),
             backgroundcolor='whitesmoke', color='black', weight='roman',
             size='12')
    plt.savefig(log_dir+'MSE_%s.pdf' %set_belong,pad_inches=0.3)
    plt.close()


def save_boxplot_MAE(losses_baseline,losses_0D,losses_2D,losses_fusion, set_belong='whole_set', log_dir='/data/titanic_1/users/sophia/myang/logs/'):
    numModels = 4
    data = [losses_baseline,losses_2D,losses_0D,losses_fusion]

    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    bp = ax1.boxplot(data, notch=0, sym='', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='blue')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.setp(bp['medians'], color = 'black')
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of MAE from different models'+'(%s)' % set_belong,fontweight="bold", size=20)
    #ax1.set_xlabel('Models')
    ax1.set_ylabel('Squared error', size=15)
    ax1.tick_params(axis = 'y', which = 'major', labelsize = 13)
    # Now fill the boxes with desired colors
    box = bp['boxes'][0]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor='lightgrey')
    ax1.add_patch(boxPolygon)

    box = bp['boxes'][1]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor='lightgreen')
    ax1.add_patch(boxPolygon)

    box = bp['boxes'][2]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor='lightblue')
    ax1.add_patch(boxPolygon)

    box = bp['boxes'][3]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor='lightcoral')
    ax1.add_patch(boxPolygon)

    ax1.set_xticklabels(['BASELINE','ONLY WIND FIELD','ONLY PAST TRACKS','FUSION NETWORK'], size='12')


    fig.text(0.15, 0.8, ' MAE = %.1f' %(np.mean(losses_baseline)),
             backgroundcolor='whitesmoke', color='black', weight='roman',
             size='12')

    fig.text(0.36, 0.8, ' MAE = %.1f' %(np.mean(losses_2D)),
             backgroundcolor='whitesmoke', color='black', weight='roman',
             size='12')
    fig.text(0.57, 0.8, ' MAE = %.1f' %(np.mean(losses_0D)),
             backgroundcolor='whitesmoke', color='black', weight='roman',
             size='12')

    fig.text(0.78, 0.8, ' MAE = %.1f' %(np.mean(losses_fusion)),
             backgroundcolor='whitesmoke', color='black', weight='roman',
             size='12')
    plt.savefig(log_dir+'MAE_%s.pdf' %set_belong,pad_inches=0.3)
    plt.close()


def draw_tracks(trainset, validset, testset, storm_id, net, criterion, criterion_mae, name='issac', windspeed_threshold=0, visual_dir= '/data/titanic_1/users/sophia/myang/logs/'):
    storm_data, set_belong = _get_data(trainset, validset, testset,storm_id)
    #print(set_belong)
    storm_data = _extract_hurricanes(storm_data,windspeed_threshold=windspeed_threshold)

    with torch.no_grad():
        input = torch.Tensor(storm_data.images).double().cuda()
        target = torch.Tensor(storm_data.labels).double().cuda()
        output = net(input, target[:,2:,:].view(target.shape[0],-1))
        loss = criterion(output, target)
        loss_mae = criterion_mae(output, target)
    squared_loss = loss.item()
    absolute_loss = loss_mae.item()
    predicted, ground_truth = output.cpu().numpy(), storm_data.labels[:,0,:]

    #aspect_ratio = abs((ground_truth[-1,0]-ground_truth[0,0])/(ground_truth[-1,1]-ground_truth[0,1]))
    plt.figure(figsize=(10, 10), dpi=150)
    plt.title('Prediction on storm track in 6 hours', fontsize=20)


    plt.scatter(ground_truth[1:-1,0]-ground_truth[:-2,0]+ground_truth[1:-1,0],ground_truth[1:-1,1]-ground_truth[:-2,1]+ground_truth[1:-1,1],
                color='grey',s=2, label='baseline')

    plt.scatter(ground_truth[1:-1,0]+predicted[1:-1,0],ground_truth[1:-1,1]+predicted[1:-1,1],
                color='coral',s=2, label='prediction')
    plt.plot(ground_truth[:,0],ground_truth[:,1], color='black',marker='o', markersize=2, linewidth=1, label='storm track' )
    for i in range(1,len(predicted)-1):
        plt.arrow(ground_truth[i,0],ground_truth[i,1],predicted[i,0],predicted[i,1],
                  color = 'lightcoral',head_width=0.1, head_length=0.1, linewidth=1.0)
    for i in range(1,len(predicted)-1):
        plt.arrow(ground_truth[i,0], ground_truth[i,1], ground_truth[i,0]-ground_truth[i-1,0],
                  ground_truth[i,1]-ground_truth[i-1,1],
                  color = 'grey',head_width=0.1, head_length=0.1, linewidth=0.5, ls='-')
    #plt.plot(ground_truth[1:,0]-ground_truth[0:-1,0]+ground_truth[1:,0],ground_truth[1:,1]-ground_truth[:-1,1]+ground_truth[1:,1], color='red',marker='o', markersize=2, linewidth=1, label='Base line' )
    #plt.plot(ground_truth[1:,0]+predicted[1:,0],ground_truth[1:,1]+predicted[1:,1], color='blue',marker='o', markersize=2, linewidth=1, label='predicted' )


    plt.ylabel('latitude')
    plt.xlabel('longtitude')
    plt.legend(prop={'size': 15})
    plt.axis('scaled')
    plt.savefig(visual_dir+name+'.pdf')
    plt.close()


def _get_data(trainset, validset, testset,storm_id):
    if storm_id in trainset.ids:
        set_belong = 'train'
        storm_data = _packinDataset(storm_id, trainset)
    elif storm_id in validset.ids:
        set_belong = 'valid'
        storm_data = _packinDataset(storm_id, validset)
    elif storm_id in testset.ids:
        set_belong = 'test'
        storm_data = _packinDataset(storm_id, testset)
    else:
        raise ValueError('storm id not found')
    return storm_data, set_belong


def _packinDataset(storm_id,dataset):
    X = dataset.images[dataset.ids == storm_id]
    Y = dataset.labels[dataset.ids == storm_id]
    ids = dataset.ids[dataset.ids == storm_id]
    timestep = np.array(dataset.timestep)[dataset.ids == storm_id]
    return MyDataset(X, Y, ids, timestep)


def _extract_hurricanes(dataset,windspeed_threshold=40):
    storm_ids = np.unique(dataset.ids)
    data = pd.read_csv("/data/titanic_1/users/sophia/sgiffard/data/Xy/2018_04_10_ERA_interim_storm/1D_data_matrix_IBTRACS.csv")
    hurricanes = data[data['windspeed'] >= windspeed_threshold]
    id = storm_ids[0]
    timestep_h = hurricanes[hurricanes['stormid'] == id]['instant_t'].values
    timestep_h = np.intersect1d(timestep_h, dataset.timestep[dataset.ids == id])
    mask = (dataset.ids == id) * (np.array([dataset.timestep[i] in timestep_h for i in range(len(dataset))]))
    X = dataset.images[mask]
    Y = dataset.labels[mask]
    ids = dataset.ids[mask]
    timestep = timestep_h
    return MyDataset(X,Y,ids,timestep)


def save_forecast_result(model, testset, testloader, log_dir, fusion=False, num_tracks=2, name='result', meta=False):
    model.eval()
    if fusion == False:
        with torch.no_grad():
            for i, (input, target) in enumerate(testloader):
                if i == 0:
                    output = model(input)
                else:
                    output = torch.cat((output, model(input)), dim=0)
    else:
        if meta == False:
            with torch.no_grad():
                for i, (input, target) in enumerate(testloader):
                    if i == 0:
                        output = model(input, target[:,-num_tracks:,:].view(target.shape[0],-1))
                    else:
                        output_ = model(input, target[:,-num_tracks:,:].view(target.shape[0],-1))
                        output = torch.cat((output, output_), dim=0)
        else:
            with torch.no_grad():
                for i, (input, target) in enumerate(testloader):
                    input_0d = target[:, -(num_tracks + 3):-1, :].view(target.shape[0], -1)
                    # add dist2land
                    dist2land = target[:, -1, 0].view(target.shape[0], -1)
                    input_0d = torch.cat((input_0d, dist2land), 1)
                    if i == 0:
                        output = model(input, input_0d)
                    else:
                        output_ = model(input, input_0d)
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


def Record_forecast(criterion_mae, log_dir, hours, filename='result.csv'):

    results = pd.read_csv(log_dir+filename).values
    output = results[:,:2].astype('float64')
    output_ids = results[:,2]
    for i in range(len(output_ids)):
        output_ids[i] = output_ids[i].strip()   #remove spaces when loading string from result.csv
    output_timesteps = results[:,3].astype('int')
    loc = results[:,4:6].astype('float64')
    ground_truth = results[:,6:8].astype('float64')
    target = np.concatenate((loc.reshape(-1,1,2), ground_truth.reshape(-1,1,2)), axis=1)

    with open('/data/titanic_1/users/sophia/myang/model/data_3d_uvz_historic6h/forecast_existing.pkl', 'rb') as f:
        data = pickle.load(f)

    for basin in ['ATL', 'EPAC']:

        bestModel_errors = [] # record their forecast model errors per year
        statisticalModel_errors = [] # record their statistical errors per year
        ourModel_errors = [] # record our prediction errors per year
        years = [] # record year which have results

        id_list = np.unique([id for id in data[basin].ids if id in output_ids])
        id_dict = {}
        for year in range(1980,2017):
            id_dict[year] = np.array([id for id in id_list if id.startswith(str(year))])

        if hours == 24:
            first_storm = True
            for year in range(1980,2017):
                if len(id_dict[year]) != 0:
                    j = 0
                    effective_timepoints = 0
                    for id in id_dict[year]:
                        '''
                        mask1: mask on 'their model' result timesteps, set True if it's in our prediction timesteps, set False if it's not in our prediction timesteps
                        mask2: mask on 'their model' result timesteps, set True if error in the timesteps is not Nan, set False if it's Nan
                        mask_forecast: mask on 'their model' result timesteps, production of mask1 and mask2, True if it's not Nan and in prediction timesteps, False if not
                        
                        all_timesteps: all timesteps which is intersection of their forecast and our prediction.
                        mask3: mask on all_timesteps, Ture if the forecast in the timestep is not Nan, False if it is Nan
                        mask_prediction: mask3
                        '''
                        mask1 = [data[basin].timestep[data[basin].ids==id][i] in output_timesteps[output_ids==id] for i in range(len(data[basin].timestep[data[basin].ids==id]))]
                        mask2 = [data[basin].M1errors[data[basin].ids==id][:,2][i] > 0 for i in range(len(data[basin].timestep[data[basin].ids==id]))]
                        mask_forecast = np.array(mask1) * np.array(mask2)
                        all_timesteps = data[basin].timestep[data[basin].ids==id][mask1]
                        mask3 = [data[basin].M1errors[data[basin].ids==id][:,2][i] > 0 for i in all_timesteps]
                        mask_prediction = np.array(mask3)
                        effective_timepoints += np.sum(mask_prediction)
                        if np.sum(mask_prediction) == 0:
                            pass
                        else:
                            if j == 0:
                                forecast_errors = data[basin].M1errors[data[basin].ids==id][:,2][mask_forecast] * 1.852
                                statistical_forecast_errors = data[basin].M2errors[data[basin].ids==id][:,2][mask_forecast] * 1.852
                                output_year = output[output_ids==id][mask_prediction]
                                target_year = target[output_ids==id][mask_prediction]
                                j = j+1
                            else:
                                forecast_errors = np.concatenate((forecast_errors, data[basin].M1errors[data[basin].ids==id][:,2][mask_forecast]* 1.852), axis=0)
                                statistical_forecast_errors = np.concatenate((statistical_forecast_errors, data[basin].M2errors[data[basin].ids==id][:,2][mask_forecast]* 1.852), axis=0)
                                output_year = np.concatenate((output_year, output[output_ids==id][mask_prediction]), axis=0)
                                target_year = np.concatenate((target_year, target[output_ids==id][mask_prediction]), axis=0)
                        outputs_onestorm = output[output_ids==id]
                        target_onestorm = target[output_ids==id]
                        if first_storm:
                            outputs_select = outputs_onestorm[mask_prediction]
                            target_select = target_onestorm[mask_prediction]
                            first_storm = False
                        else:
                            outputs_select = np.concatenate((outputs_select, outputs_onestorm[mask_prediction]), axis=0)
                            target_select = np.concatenate((target_select, target_onestorm[mask_prediction]), axis=0)

                        if year>=2014:
                            predicted_errors = np.array([loss_notorch.Regress_Loss_Mae_notorch(outputs_onestorm[i:i+1], target_onestorm[i:i+1])
                                                for i in range(len(outputs_onestorm))])
                            forecast_errors_1 = data[basin].M1errors[data[basin].ids==id][:,2] * 1.852
                            forecast_errors_2 = data[basin].M2errors[data[basin].ids==id][:,2] * 1.852
                            num_prefix = output_timesteps[output_ids==id][0]
                            num_postfix = data[basin].timestep[data[basin].ids==id][-1] - output_timesteps[output_ids==id][-1]
                            prefix = np.array([None for i in range(num_prefix)])
                            postfix = np.array([None for i in range(num_postfix)])
                            predicted_errors = np.concatenate((prefix, predicted_errors, postfix), axis=0)
                            xs = np.arange(len(forecast_errors_1))
                            plt.figure(figsize=(10, 8))
                            plt.title(id,fontweight="bold", size=20)
                            plt.plot(xs, predicted_errors,linestyle='-', color='blue', label='predicted errors')
                            plt.plot(xs, forecast_errors_1,linestyle='-', color='red', label = 'best model forcast errors')
                            plt.plot(xs, forecast_errors_2,linestyle='-', color='green', label = 'statistical model forcast errors')
                            plt.xlabel('timesteps', size=15)
                            plt.ylabel('Errors(km)', size=15)
                            my_x_ticks=np.arange(0, len(forecast_errors_1),1)
                            plt.xticks(my_x_ticks)
                            plt.legend()
                            plt.savefig(log_dir+'errors_comparison_'+id+'.pdf')
                            plt.close()

                    if j==0:
                        with open(log_dir + 'Res_models_tracks_per_year_'+basin+'.txt', 'a+') as f:
                            f.write('year: %s \t mean forecast error: %s \t mean predicted error: %s \n' % (str(year), 'nan', 'nan'))
                        continue
                    fe_mean = np.mean(forecast_errors)
                    sfe_mean = np.mean(statistical_forecast_errors)
                    pe_mean = loss_notorch.Regress_Loss_Mae_notorch(output_year, target_year)
                    with open(log_dir + 'Res_models_tracks_per_year_'+basin+'.txt', 'a+') as f:
                        f.write('year: %s \t mean M2 forecast error: %.2f \t mean predicted error: %.2f \t mean M1 forecast error in year: %.2f \t '
                                'number of storms in year: %d \t number of timepoints: %d \n' % (str(year), fe_mean, pe_mean, sfe_mean,
                                                                                                                                   len(id_dict[year]), effective_timepoints))
                    '''
                    update records per year
                    '''
                    bestModel_errors.append(fe_mean)
                    statisticalModel_errors.append(sfe_mean)
                    ourModel_errors.append(pe_mean)
                    years.append(year)

                else:
                    with open(log_dir + 'Res_models_tracks_per_year_'+basin+'.txt', 'a+') as f:
                            f.write('year: %s \t mean forecast error: %s \t mean predicted error: %s \n' % (str(year), 'nan', 'nan'))

            statistical_forecast_errors = data[basin].M1errors[:,2]
            statistical_forecast_errors = statistical_forecast_errors[~np.isnan(statistical_forecast_errors)]* 1.852
            mean_statistical_forecast_errors = np.mean(statistical_forecast_errors)
            std_statistical_forecast_errors = np.std(statistical_forecast_errors)
            mean_predicted_forecast_errors = loss_notorch.Regress_Loss_Mae_notorch(outputs_select, target_select)
            errors = []
            for i in range(len(outputs_select)):
                errors.append(loss_notorch.Regress_Loss_Mae_notorch(outputs_select[i:i+1], target_select[i:i+1]))
            std_predicted_forecast_errors = np.std(np.array(errors))
            with open(log_dir + 'Res_models_tracks_per_year_'+basin+'.txt', 'a+') as f:
                f.write('mean_statistical_forecast_errors: %.5f \t std_statistical_forecast_errors: %.5f \n '
                        'mean_predicted_forecast_errors: %.5f \t std_predicted_forecast_errors:  %.5f \n'
                        % (mean_statistical_forecast_errors, std_statistical_forecast_errors, mean_predicted_forecast_errors, std_predicted_forecast_errors))


            '''
            trend plot of errors per year
            '''
            plt.figure(figsize=(20, 16))
            plt.title('comparison of models errors per year-' + basin,fontweight="bold", size=20)
            xs = years
            plt.plot(xs, ourModel_errors,linestyle='-', color='blue', label='predicted errors')
            plt.plot(xs, bestModel_errors,linestyle='-', color='red', label = 'best model forcast errors')
            plt.plot(xs, statisticalModel_errors,linestyle='-', color='green', label = 'statistical model forcast errors')
            plt.xlabel('years', size=20)
            plt.ylabel('Errors(km)', size=20)
            my_x_ticks=np.arange(years[0], years[-1]+1,2)
            plt.xticks(my_x_ticks)
            plt.legend(prop={'size': 20})
            plt.savefig(log_dir+basin+'_comparison_of_models_errors_per_year.pdf')
            plt.show()
            plt.close()


def plot_forecast_compare(log_dir, hours):
    '''
    this function plots the errors along the tracks as bars (2014 to 2016 only),
    and compare it to the two model forecasts.
    :param log_dir: the results directory
    :param hours: should be 6, 24 or another factor of 6.
    :return: 0 (plots figures)
    '''

    # current results :
    Data_pred = pd.read_csv(log_dir+'result.csv', header=None)
    pred_uv = np.array([Data_pred[0], Data_pred[1]]).T
    stormid_test = np.array([x.strip(' ') for x in Data_pred[2]]) #storms in the test set (or valid)
    instant_t = np.array(Data_pred[3])
    coords_t = np.array([Data_pred[4], Data_pred[5]]).T

    folder_comp='/data/titanic_1/users/sophia/sgiffard/data/forecasts_existing/'
    for basin in ['ATL', 'EPAC']:
        # model results (atlantic and pacific only)
        base_namefile=folder_comp+'1989-present_OFCL_v_BCD5_ind_'+basin +'_AC_errors'
        with open(base_namefile + 'processed_longlat_ournamings.pkl', 'rb') as pickle_file:
            Data = pickle.load(pickle_file)
        f_times_new = Data['f_times']
        f_lats_new = Data['f_lats']
        f_longs_new = Data['f_longs']
        forecast_m1_lon_new = Data['forecast_m1_lon_new']
        forecast_m1_lat_new = Data['forecast_m1_lat_new']
        forecast_m2_lon_new = Data['forecast_m2_lon_new']
        forecast_m2_lat_new = Data['forecast_m2_lat_new']
        hours_forecasts = Data['hours_forecasts']
        models = Data['models']

        # take only 2014,2015 and 2016 stormids
        stormids=f_times_new.keys()
        stormids=[id for id in stormids if id[:4] in ['2014','2015','2016']]
        h_ind=list(hours_forecasts).index(hours)

        for stormid  in stormids:

            if stormid not in stormid_test: #can't compare results of train set...!
                continue

            lats = f_lats_new[stormid]
            longs = f_longs_new[stormid]
            true_coords=np.array([longs,lats])

            m1_lat_h=[]; m1_lon_h=[]; m2_lat_h=[]; m2_lon_h=[]
            length_storm=len(forecast_m1_lat_new[stormid])
            for i in range(length_storm):
                m1_lat_h.append(forecast_m1_lat_new[stormid][i][h_ind])
                m1_lon_h.append(forecast_m1_lon_new[stormid][i][h_ind])
                m2_lat_h.append(forecast_m2_lat_new[stormid][i][h_ind])
                m2_lon_h.append(forecast_m2_lon_new[stormid][i][h_ind])

            # get the same storms, and add nans when there are no possibe comparison.
            pred_uv_storm_dep=pred_uv[stormid_test==stormid]
            coords_t_storm_dep=coords_t[stormid_test == stormid]
            instant_t_storm=instant_t[stormid_test == stormid]
            pred_uv_storm=[]; coords_t_storm=[]; pred_lonlat=[]
            for i in range(length_storm):
                ind=np.where(instant_t_storm ==i)[0]
                if np.isnan(m1_lat_h[i]) or not ind:
                    pred_uv_storm.append([np.nan,np.nan])
                    coords_t_storm.append([np.nan,np.nan])
                    pred_lonlat.append([np.nan,np.nan])
                else:
                    curr_uv=pred_uv_storm_dep[ind[0]]
                    pred_uv_storm.append(curr_uv)
                    coords_t_storm.append(coords_t_storm_dep[ind[0]])
                    pred_lonlat.append([longs[i]+curr_uv[0],lats[i]+curr_uv[1]])
            pred_lonlat=np.array(pred_lonlat)

            plt.figure(dpi=400,figsize=(10,4))
            plt.subplot(1, 3, 1)
            plt.plot(true_coords[0], true_coords[1], color='black', marker='o', markersize=2, linewidth=1)
            for i in range(length_storm - 4):
                plt.plot([m1_lon_h[i],  longs[i + 4]], [m1_lat_h[i], lats[i + 4]], color='red')
            plt.xlabel('model ' + models[0] + ' (official)')

            plt.subplot(1, 3, 2)
            plt.plot(true_coords[0], true_coords[1], color='black', marker='o', markersize=2, linewidth=1)
            for i in range(length_storm - 4):
                plt.plot([m2_lon_h[i], longs[i + 4]], [m2_lat_h[i], lats[i + 4]], color='green')
            plt.xlabel('model ' + models[1] + ' (statistical)' )

            plt.subplot(1, 3, 3)
            plt.plot(true_coords[0], true_coords[1], color='black', marker='o', markersize=2, linewidth=1)
            for i in range(length_storm-4):
                plt.plot([pred_lonlat[i][0], longs[i + 4]], [pred_lonlat[i][1], lats[i + 4]], color='blue')
            plt.xlabel('Deep NN')

            plt.suptitle(str(hours)+'-h forecast trajectory errors '+str(stormid))
            plt.savefig(log_dir +'res_storm_bars_subplots_results_'+str(stormid)+'.png')
            plt.savefig(log_dir + 'res_storm_bars_subplots_results_' + str(stormid) + '.pdf')
            plt.close()

    return 0
