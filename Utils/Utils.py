## Module to provide tools for deep learning
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from Utils.MyDataset import MyDataset

DIR = "/data/titanic_1/users/sophia/myang/model/"

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, checkpoint_dir=DIR, name='model_best.pth.tar', filename_temp='checkpoint.pth.tar'):
    """
    save checkpoint at the end of epoch, if the checkpoint have the best precision, copy to 'model_best.pth.tar'
    :param state: parameters of the model
    :param is_best: boolean variable which indicate if the model have the best precision
    :param checkpoint_dir: directory to save the checkpoint
    :return: None
    """
    filename=checkpoint_dir+filename_temp
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir+name)

def early_stopping(train_loss, patience=10, min_delta=1.0):
    '''
    only one mode for loss: the lower the better
    :param train_loss: list of train loss in every epochs
    :param patience: the number of epochs who don't improve to trigger early stopping
    :param min_delta:
    :return: None
    '''
    if len(train_loss) <= patience:
        return False
    elif max(train_loss[-patience:]) - min(train_loss[-patience:]) <= min_delta:
        return True
    else:
        return False

def dataset_filter(trainset, validset, testset, hours, with_tracks=False):
    """
    reform the dataset for different tasks(number of hours for prediction, with or without tracks)
    :param trainset: trainset in format of MyDataset
    :param validset: validset in format of MyDataset
    :param testset: testset in format of MyDataset
    :param hours: number of hours, choices from [6,12] for now
    :param with_tracks: if True, tracks information will be included
    :return: trainset, validset, testset in format of MyDataset
    """




    # if the dataset should include tracks, do following
    if with_tracks:
        for dataset in (trainset, validset, testset):
            list_storms = np.unique(dataset.ids) #get all ids of storm
            del_list = [] #get the indexs of examples to be deleted
            newset = np.zeros([dataset.labels.shape[0],6,2]) #initiate newset with the shape wanted to replace dataset.labels
            for storm in list_storms:
                storm_idx = np.where(dataset.ids==storm)[0] # get all indexs of examples in one storm
                newset[:,:2,:] = dataset.labels[:,:2,:] #copy the first two colomns
                newset[storm_idx[1:],2,:] = dataset.labels[storm_idx[:-1],1,:] - dataset.labels[storm_idx[:-1],0,:] #set 3rd column to be t-1 deplacement
                newset[storm_idx[2:],3,:] = dataset.labels[storm_idx[:-2],1,:] - dataset.labels[storm_idx[:-2],0,:] #set 4rd column to be t-2 deplacement
                newset[storm_idx[3:],4,:] = dataset.labels[storm_idx[:-3],1,:] - dataset.labels[storm_idx[:-3],0,:] #set 5rd column to be t-3 deplacement
                newset[storm_idx[4:],5,:] = dataset.labels[storm_idx[:-4],1,:] - dataset.labels[storm_idx[:-4],0,:] #set 6rd column to be t-4 deplacement
                del_list.extend(storm_idx[:4]) # append the indexs of the first 4 examples to del_list (knowing that they don'y have all 4 history tracks)
            # delete examples in del_list
            newset = np.delete(newset, del_list, axis = 0)
            dataset.labels = newset
            dataset.images = np.delete(dataset.images, del_list, axis = 0)
            dataset.ids = np.delete(dataset.ids, del_list, axis = 0)

    # if the task is to predict after 6 hours, just return all dataset
    if hours == 6:
        return (trainset, validset, testset)

    # if the task is to predict after 6 hours, reshape to get the correct label and return all dataset
    if hours == 12:
        for dataset in (trainset, validset, testset):
            list_storms = np.unique(dataset.ids)
            del_list = []
            for storm in list_storms:
                storm_idx = np.where(dataset.ids==storm)[0]
                # only move forward the second column (next coordinates), keep all other columns
                dataset.labels[storm_idx[:-1],1,:] = dataset.labels[storm_idx[1:],1,:]
                # add the index of the last example to del_list (which don't have prediction after 12 h)
                del_list.append(storm_idx[-1])
            dataset.labels = np.delete(dataset.labels, del_list, axis = 0)
            dataset.images = np.delete(dataset.images, del_list, axis = 0)
            dataset.ids = np.delete(dataset.ids, del_list, axis = 0)

        return (trainset, validset, testset)

def extract_hurricanes(dataset):
    storm_ids = np.unique(dataset.ids)
    data = pd.read_csv("/data/titanic_1/users/sophia/sgiffard/data/Xy/2018_04_10_ERA_interim_storm/1D_data_matrix_IBTRACS.csv")
    hurricanes = data[data['windspeed'] >= 64]
    first = True
    for id in tqdm(storm_ids):
        timestep_h = hurricanes[hurricanes['stormid'] == id]['instant_t'].values
        if len(timestep_h) == 0:
            continue
        timestep_h = np.intersect1d(timestep_h, dataset.timestep[dataset.ids == id])
        mask = (dataset.ids == id) * (np.array([dataset.timestep[i] in timestep_h for i in range(len(dataset))]))
        if first:
            X = dataset.images[mask]
            Y = dataset.labels[mask]
            ids = dataset.ids[mask]
            timestep = timestep_h
            first=False
        else:
            X = np.concatenate((X,dataset.images[mask]), axis=0)
            Y = np.concatenate((Y,dataset.labels[mask]), axis=0)
            ids = np.concatenate((ids, dataset.ids[mask]), axis=0)
            timestep = np.concatenate((timestep, timestep_h), axis=0)
    return MyDataset(X,Y,ids,timestep)

def get_baselineset(dataset, hours = 24):
    indexes = np.unique(dataset.ids, return_index=True)[1]
    ids_list = [dataset.ids[index] for index in sorted(indexes)]

    ground_truth = []
    last_disp = []
    ids = []
    timesteps = []

    for id in tqdm(ids_list):
        num_storm = len(dataset.labels[dataset.ids == id][1:])
        if hours == 6:
            last_disp.extend(dataset.labels[dataset.ids == id][:-1,1,:] - dataset.labels[dataset.ids == id][:-1,0,:])
        elif hours == 12:
            last_disp.extend(dataset.labels[dataset.ids == id][:-1,2,:] - dataset.labels[dataset.ids == id][:-1,0,:])
        elif hours == 18:
            last_disp.extend(dataset.labels[dataset.ids == id][:-1,3,:] - dataset.labels[dataset.ids == id][:-1,0,:])
        elif hours == 24:
            last_disp.extend(dataset.labels[dataset.ids == id][:-1,4,:] - dataset.labels[dataset.ids == id][:-1,0,:])
        ground_truth.extend(dataset.labels[dataset.ids == id][1:])
        ids.extend([id]*num_storm)
        timesteps.extend(list(range(1,num_storm+1)))


    return MyDataset(np.array(last_disp),np.array(ground_truth), np.array(ids),np.array(timesteps))

def dataset_filter_3D(trainset, validset, testset, hours, with_tracks=False,num_tracks=2, with_windspeed=False, normalize=True, get_derivative_xy=False, levels=[6], params = [0, 1, 3, 7, 9, 10, 11, 12, 13]):
    """
    reform the dataset for different tasks(number of hours for prediction, with or without tracks)
    :param trainset: trainset in format of MyDataset
    :param validset: validset in format of MyDataset
    :param testset: testset in format of MyDataset
    :param hours: number of hours, choices from [6,12] for now
    :param with_tracks: if True, tracks information will be included
    :return: trainset, validset, testset in format of MyDataset

    for example, if hours = 24, with_tracks=2, with_windspeed=True:
    target shape: [n, c, 2]
    n: number of example
    c: number of channels
    2: each channel contains 2 values

    c will be equal to 8 different channels:
    0: lon, lat for current time step (always)
    1: lon, lat after 24 hours (24h)
    2: lon. lat after 18 hours (24h)
    3: lon, lat after 12 hours (24h)
    4: lon, lat after 6 hours (24h)
    5: t-1 to t deplacement (with 2 tracks)
    6: t-2 to t-1 deplacement (with 2 tracks)
    7: windspeed and Jdays (with windspeed)
    8: long lat standardized (with windspeed)
    9: dist2land and 0 (with windspeed)


    """
    #original_levlist=['125','175','225','300','400','500','600','700','775','825','900','950','1000']
    #original param list=['r', 'd', 'o3', 'v', 'ciwc', 'q', 'pv', 'z', 'clwc', 't', 'w', 'vo', 'u', 'cc']
    #useful param list = ['r', 'u','v','w','vo','d','t','z','cc']
    #param filter = [0, 1, 3, 7, 9, 10, 11, 12, 13]



    # if the dataset should include tracks, do following
    if with_tracks:
        metadata = pd.read_csv("/data/titanic_1/users/sophia/sgiffard/data/Xy/2018_04_10_ERA_interim_storm/1D_data_matrix_IBTRACS.csv") # all 0D data

        if with_windspeed:
            # standardize the 0d data (use only train samples to get mean and std). then change data.
            list_storms = np.unique(trainset.ids)
            winds=[]; longs=[]; lats=[]; d2l=[]; Jdp=[]
            for storm in list_storms:
                winds.extend(metadata['windspeed'][metadata['stormid'] == storm].values)
                longs.extend(metadata['longitude'][metadata['stormid'] == storm].values)
                lats.extend(metadata['latitude'][metadata['stormid'] == storm].values)
                Jdp.extend(metadata['Jday_predictor'][metadata['stormid'] == storm].values)
                d2l.extend(metadata['dist2land'][metadata['stormid'] == storm].values)
            metadata['windspeed']=(metadata['windspeed']-np.mean(winds))/np.std(winds)
            metadata['longitude'] = (metadata['longitude'] - np.mean(longs)) / np.std(longs)
            metadata['latitude'] = (metadata['latitude'] - np.mean(lats)) / np.std(lats)
            metadata['Jday_predictor'] = (metadata['Jday_predictor'] - np.mean(Jdp)) / np.std(Jdp)
            metadata['dist2land'] = (metadata['dist2land'] - np.mean(d2l)) / np.std(d2l)



        for dataset in (trainset, validset, testset):
            list_storms = np.unique(dataset.ids) #get all ids of storm
            del_list = [] #get the indexs of examples to be deleted
            if num_tracks == 0:
                raise ValueError("number of tracks must be more or equal to 1")

            if num_tracks == 1:
                if with_windspeed == False:
                    newset = np.zeros([dataset.labels.shape[0],3,2]) #initiate newset with the shape wanted to replace dataset.labels
                    newset[:,:2,:] = dataset.labels[:,:2,:] #copy the first two colomns
                    for storm in list_storms:
                        storm_idx = np.where(dataset.ids==storm)[0] # get all indexs of examples in one storm
                        newset[storm_idx[1:],2,:] = dataset.labels[storm_idx[:-1],1,:] - dataset.labels[storm_idx[:-1],0,:] #set 3rd column to be t-1 deplacement
                        del_list.extend(storm_idx[:1]) # append the indexs of the first 4 examples to del_list (knowing that they don'y have all 4 history tracks)
                else:
                    newset = np.zeros([dataset.labels.shape[0],6,2]) #initiate newset with the shape wanted to replace dataset.labels
                    newset[:,:2,:] = dataset.labels[:,:2,:] #copy the first two colomns
                    for storm in list_storms:
                        storm_idx = np.where(dataset.ids==storm)[0] # get all indexs of examples in one storm
                        num_datapoints = len(storm_idx)
                        newset[storm_idx[1:],2,:] = dataset.labels[storm_idx[:-1],1,:] - dataset.labels[storm_idx[:-1],0,:] #set 3rd column to be t-1 deplacement
                        newset[storm_idx[:],3,0] = metadata['windspeed'][metadata['stormid'] == storm].values[:num_datapoints]
                        newset[storm_idx[:],3,1] = metadata['Jday_predictor'][metadata['stormid'] == storm].values[:num_datapoints]
                        newset[storm_idx[:], 4, 0] = metadata['longitude'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        newset[storm_idx[:], 4, 1] = metadata['latitude'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        newset[storm_idx[:], 5, 0] = metadata['dist2land'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        del_list.extend(storm_idx[:1]) # append the indexs of the first 4 examples to del_list (knowing that they don'y have all 4 history tracks)

            elif num_tracks == 2:
                if with_windspeed == False:
                    newset = np.zeros([dataset.labels.shape[0],4,2]) #initiate newset with the shape wanted to replace dataset.labels
                    newset[:,:2,:] = dataset.labels[:,:2,:] #copy the first two colomns
                    for storm in list_storms:
                        storm_idx = np.where(dataset.ids==storm)[0] # get all indexs of examples in one storm
                        newset[storm_idx[1:],2,:] = dataset.labels[storm_idx[:-1],1,:] - dataset.labels[storm_idx[:-1],0,:] #set 3rd column to be t-1 deplacement
                        newset[storm_idx[2:],3,:] = dataset.labels[storm_idx[:-2],1,:] - dataset.labels[storm_idx[:-2],0,:] #set 4rd column to be t-2 deplacement
                        del_list.extend(storm_idx[:2]) # append the indexs of the first 4 examples to del_list (knowing that they don'y have all 4 history tracks)
                else:
                    newset = np.zeros([dataset.labels.shape[0],7,2]) #initiate newset with the shape wanted to replace dataset.labels
                    newset[:,:2,:] = dataset.labels[:,:2,:] #copy the first two colomns
                    for storm in list_storms:
                        storm_idx = np.where(dataset.ids==storm)[0] # get all indexs of examples in one storm
                        num_datapoints = len(storm_idx)
                        newset[storm_idx[1:],2,:] = dataset.labels[storm_idx[:-1],1,:] - dataset.labels[storm_idx[:-1],0,:] #set 3rd column to be t-1 deplacement
                        newset[storm_idx[2:],3,:] = dataset.labels[storm_idx[:-2],1,:] - dataset.labels[storm_idx[:-2],0,:] #set 4rd column to be t-2 deplacement
                        newset[storm_idx[:],4,0] = metadata['windspeed'][metadata['stormid'] == storm].values[:num_datapoints] #set 5rd column to be windspeed of current time
                        newset[storm_idx[:],4,1] = metadata['Jday_predictor'][metadata['stormid'] == storm].values[:num_datapoints]
                        newset[storm_idx[:], 5, 0] = metadata['longitude'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        newset[storm_idx[:], 5, 1] = metadata['latitude'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        newset[storm_idx[:], 6, 0] = metadata['dist2land'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]

                        del_list.extend(storm_idx[:2]) # append the indexs of the first 4 examples to del_list (knowing that they don'y have all 4 history tracks)

            elif num_tracks == 3:
                if with_windspeed == False:
                    newset = np.zeros([dataset.labels.shape[0],5,2]) #initiate newset with the shape wanted to replace dataset.labels
                    newset[:,:2,:] = dataset.labels[:,:2,:] #copy the first two colomns
                    for storm in list_storms:
                        storm_idx = np.where(dataset.ids==storm)[0] # get all indexs of examples in one storm
                        newset[storm_idx[1:],2,:] = dataset.labels[storm_idx[:-1],1,:] - dataset.labels[storm_idx[:-1],0,:] #set 3rd column to be t-1 deplacement
                        newset[storm_idx[2:],3,:] = dataset.labels[storm_idx[:-2],1,:] - dataset.labels[storm_idx[:-2],0,:] #set 4rd column to be t-2 deplacement
                        newset[storm_idx[3:],4,:] = dataset.labels[storm_idx[:-3],1,:] - dataset.labels[storm_idx[:-3],0,:] #set 5rd column to be t-3 deplacement
                        del_list.extend(storm_idx[:3]) # append the indexs of the first 4 examples to del_list (knowing that they don'y have all 4 history tracks)
                else:
                    newset = np.zeros([dataset.labels.shape[0],8,2]) #initiate newset with the shape wanted to replace dataset.labels
                    newset[:,:2,:] = dataset.labels[:,:2,:] #copy the first two colomns
                    for storm in list_storms:
                        storm_idx = np.where(dataset.ids==storm)[0] # get all indexs of examples in one storm
                        num_datapoints = len(storm_idx)
                        newset[storm_idx[1:],2,:] = dataset.labels[storm_idx[:-1],1,:] - dataset.labels[storm_idx[:-1],0,:] #set 3rd column to be t-1 deplacement
                        newset[storm_idx[2:],3,:] = dataset.labels[storm_idx[:-2],1,:] - dataset.labels[storm_idx[:-2],0,:] #set 4rd column to be t-2 deplacement
                        newset[storm_idx[3:],4,:] = dataset.labels[storm_idx[:-3],1,:] - dataset.labels[storm_idx[:-3],0,:] #set 5rd column to be t-3 deplacement
                        newset[storm_idx[:],5,0] = metadata['windspeed'][metadata['stormid'] == storm].values[:num_datapoints]
                        newset[storm_idx[:],5,1] = metadata['Jday_predictor'][metadata['stormid'] == storm].values[:num_datapoints]
                        newset[storm_idx[:], 6, 0] = metadata['longitude'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        newset[storm_idx[:], 6, 1] = metadata['latitude'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        newset[storm_idx[:], 7, 0] = metadata['dist2land'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        del_list.extend(storm_idx[:3]) # append the indexs of the first 4 examples to del_list (knowing that they don'y have all 4 history tracks)

            elif num_tracks == 4:
                if with_windspeed == False:
                    newset = np.zeros([dataset.labels.shape[0],6,2]) #initiate newset with the shape wanted to replace dataset.labels
                    newset[:,:2,:] = dataset.labels[:,:2,:] #copy the first two colomns
                    for storm in list_storms:
                        storm_idx = np.where(dataset.ids==storm)[0] # get all indexs of examples in one storm
                        newset[storm_idx[1:],2,:] = dataset.labels[storm_idx[:-1],1,:] - dataset.labels[storm_idx[:-1],0,:] #set 3rd column to be t-1 deplacement
                        newset[storm_idx[2:],3,:] = dataset.labels[storm_idx[:-2],1,:] - dataset.labels[storm_idx[:-2],0,:] #set 4rd column to be t-2 deplacement
                        newset[storm_idx[3:],4,:] = dataset.labels[storm_idx[:-3],1,:] - dataset.labels[storm_idx[:-3],0,:] #set 5rd column to be t-3 deplacement
                        newset[storm_idx[4:],5,:] = dataset.labels[storm_idx[:-4],1,:] - dataset.labels[storm_idx[:-4],0,:] #set 6rd column to be t-4 deplacement
                        del_list.extend(storm_idx[:4]) # append the indexs of the first 4 examples to del_list (knowing that they don'y have all 4 history tracks)
                else:
                    newset = np.zeros([dataset.labels.shape[0],9,2]) #initiate newset with the shape wanted to replace dataset.labels
                    newset[:,:2,:] = dataset.labels[:,:2,:] #copy the first two colomns
                    for storm in list_storms:
                        storm_idx = np.where(dataset.ids==storm)[0] # get all indexs of examples in one storm
                        num_datapoints = len(storm_idx)
                        newset[storm_idx[1:],2,:] = dataset.labels[storm_idx[:-1],1,:] - dataset.labels[storm_idx[:-1],0,:] #set 3rd column to be t-1 deplacement
                        newset[storm_idx[2:],3,:] = dataset.labels[storm_idx[:-2],1,:] - dataset.labels[storm_idx[:-2],0,:] #set 4rd column to be t-2 deplacement
                        newset[storm_idx[3:],4,:] = dataset.labels[storm_idx[:-3],1,:] - dataset.labels[storm_idx[:-3],0,:] #set 5rd column to be t-3 deplacement
                        newset[storm_idx[4:],5,:] = dataset.labels[storm_idx[:-4],1,:] - dataset.labels[storm_idx[:-4],0,:] #set 6rd column to be t-4 deplacement
                        newset[storm_idx[:],6,0] = metadata['windspeed'][metadata['stormid'] == storm].values[:num_datapoints]
                        newset[storm_idx[:],6,1] = metadata['Jday_predictor'][metadata['stormid'] == storm].values[:num_datapoints]
                        newset[storm_idx[:], 7, 0] = metadata['longitude'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        newset[storm_idx[:], 7, 1] = metadata['latitude'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]
                        newset[storm_idx[:], 8, 0] = metadata['dist2land'][metadata['stormid'] == storm].values[
                                                     :num_datapoints]

                        del_list.extend(storm_idx[:4]) # append the indexs of the first 4 examples to del_list (knowing that they don'y have all 4 history tracks)
            else:
                raise ValueError("number of tracks more than 4 are currently not supported")


            # delete examples in del_list
            newset = np.delete(newset, del_list, axis = 0)

            # replace the displacements in -180,180 degrees.
            if not with_windspeed:
                newset[:,2:,:][newset[:,2:,:]<-180] = newset[:,2:,:][newset[:,2:,:]<-180] + 360
                newset[:,2:,:][newset[:,2:,:]>180] = newset[:,2:,:][newset[:,2:,:]>180] - 360
            else:
                newset[:,2:-4,:][newset[:,2:-4,:]<-180] = newset[:,2:-4,:][newset[:,2:-4,:]<-180] + 360
                newset[:,2:-4,:][newset[:,2:-4,:]>180] = newset[:,2:-4,:][newset[:,2:-4,:]>180] - 360

            dataset.labels = newset
            dataset.images = np.delete(dataset.images, del_list, axis = 0)
            dataset.ids = np.delete(dataset.ids, del_list, axis = 0)
            dataset.timestep = np.delete(np.array(dataset.timestep), del_list, axis=0)

    else:
        if num_tracks!=0:
            raise ValueError("number of tracks should be 0!")


    # if the task is to predict after 6 hours, just return all dataset
    if hours == 6:
        for dataset in (trainset, validset, testset):
            list_storms = np.unique(dataset.ids)
            del_list = []
            for storm in list_storms:
                storm_idx = np.where(dataset.ids==storm)[0]
                # add the index of the last example to del_list (which don't have last deplacement between 6h ago and now)
                del_list.extend(storm_idx[0:1])
            dataset.labels = np.delete(dataset.labels, del_list, axis = 0)
            dataset.images = np.delete(dataset.images, del_list, axis = 0)
            dataset.ids = np.delete(dataset.ids, del_list, axis = 0)
            dataset.timestep = np.delete(np.array(dataset.timestep), del_list, axis=0)

    # if the task is to predict after 6 hours, reshape to get the correct label and return all dataset
    elif hours == 12:
        for dataset in (trainset, validset, testset):
            list_storms = np.unique(dataset.ids)
            del_list = []
            for storm in list_storms:
                storm_idx = np.where(dataset.ids==storm)[0]
                # only move forward the second column (next coordinates), keep all other columns
                dataset.labels[storm_idx[:-1],1,:] = dataset.labels[storm_idx[1:],1,:]
                # add the index of the last example to del_list (which don't have prediction after 24 h)
                del_list.extend(storm_idx[-1])
                del_list.extend(storm_idx[:1])
            dataset.labels = np.delete(dataset.labels, del_list, axis = 0)
            dataset.images = np.delete(dataset.images, del_list, axis = 0)
            dataset.ids = np.delete(dataset.ids, del_list, axis = 0)
            dataset.timestep = np.delete(np.array(dataset.timestep), del_list, axis=0)

    elif hours == 24:
        for dataset in (trainset, validset, testset):
            list_storms = np.unique(dataset.ids)
            newset = np.zeros([dataset.labels.shape[0],4,2]) #initiate newset with the shape wanted to replace dataset.labels
            del_list = []
            for storm in list_storms:
                storm_idx = np.where(dataset.ids==storm)[0]
                # only move forward the second column (next coordinates), keep all other columns
                #dataset.labels[storm_idx[:-3],1,:] = dataset.labels[storm_idx[3:],1,:]
                newset[storm_idx[:-3],0,:] = dataset.labels[storm_idx[3:],1,:]
                newset[storm_idx[:-2],1,:] = dataset.labels[storm_idx[2:],1,:]
                newset[storm_idx[:-1],2,:] = dataset.labels[storm_idx[1:],1,:]
                newset[storm_idx[:],3,:] = dataset.labels[storm_idx[:],1,:]

                # add the index of the last example to del_list (which don't have prediction after 24 h)
                del_list.extend(storm_idx[-3:])
                del_list.extend(storm_idx[:1])
            #dataset.labels = np.delete(dataset.labels, 1, axis=1)
            dataset.labels = np.concatenate((dataset.labels[:,0:1,:], newset, dataset.labels[:,2:,:]), axis=1)
            dataset.labels = np.delete(dataset.labels, del_list, axis = 0)
            dataset.images = np.delete(dataset.images, del_list, axis = 0)
            dataset.ids = np.delete(dataset.ids, del_list, axis = 0)
            dataset.timestep = np.delete(np.array(dataset.timestep), del_list, axis=0)


    elif hours == 48:
        for dataset in (trainset, validset, testset):
            list_storms = np.unique(dataset.ids)
            newset = np.zeros([dataset.labels.shape[0],8,2]) #initiate newset with the shape wanted to replace dataset.labels
            del_list = []
            for storm in list_storms:
                storm_idx = np.where(dataset.ids==storm)[0]
                # only move forward the second column (next coordinates), keep all other columns
                #dataset.labels[storm_idx[:-3],1,:] = dataset.labels[storm_idx[3:],1,:]
                newset[storm_idx[:-7], 0, :] = dataset.labels[storm_idx[7:], 1, :]
                newset[storm_idx[:-6], 1, :] = dataset.labels[storm_idx[6:], 1, :]
                newset[storm_idx[:-5], 2, :] = dataset.labels[storm_idx[5:], 1, :]
                newset[storm_idx[:-4], 3, :] = dataset.labels[storm_idx[4:], 1, :]
                newset[storm_idx[:-3],4,:] = dataset.labels[storm_idx[3:],1,:]
                newset[storm_idx[:-2],5,:] = dataset.labels[storm_idx[2:],1,:]
                newset[storm_idx[:-1],6,:] = dataset.labels[storm_idx[1:],1,:]
                newset[storm_idx[:],7,:] = dataset.labels[storm_idx[:],1,:]

                # add the index of the last example to del_list (which don't have prediction after 24 h)
                del_list.extend(storm_idx[-7:])
                del_list.extend(storm_idx[:1])
            #dataset.labels = np.delete(dataset.labels, 1, axis=1)
            dataset.labels = np.concatenate((dataset.labels[:,0:1,:], newset, dataset.labels[:,2:,:]), axis=1)
            dataset.labels = np.delete(dataset.labels, del_list, axis = 0)
            dataset.images = np.delete(dataset.images, del_list, axis = 0)
            dataset.ids = np.delete(dataset.ids, del_list, axis = 0)
            dataset.timestep = np.delete(np.array(dataset.timestep), del_list, axis=0)

    if len(params) != 0:
        for dataset in (trainset, validset, testset):
            dataset.images = dataset.images[:,:,levels,:,:]
            dataset.images = dataset.images[:,params,:,:,:]


        if len(levels) == 1 and len(params)==1:
            for dataset in (trainset, validset, testset):
                dataset.images = dataset.images.reshape(dataset.images.shape[0], dataset.images.shape[3], dataset.images.shape[4])

        elif len(levels) == 1 and len(params) > 1:
            for dataset in (trainset, validset, testset):
                dataset.images = dataset.images.reshape(dataset.images.shape[0], dataset.images.shape[1], dataset.images.shape[3], dataset.images.shape[4])

        elif len(params) == 1 and len(levels) > 1:
            for dataset in (trainset, validset, testset):
                dataset.images = dataset.images.reshape(dataset.images.shape[0], dataset.images.shape[2], dataset.images.shape[3], dataset.images.shape[4])

    else:
        pass

    if normalize==True:
        trainset, validset, testset = _normalize(trainset, validset, testset)

    if get_derivative_xy == True:
        trainset, validset, testset = _get_derivative_xy(trainset, validset, testset)

    return (trainset, validset, testset)





def _normalize(trainset, validset, testset):
    input_dim = len(trainset.images.shape)
    num_channels = trainset.images.shape[1]
    means = np.zeros(num_channels)
    stds = np.zeros(num_channels)
    if input_dim == 5:
        for i in range(means.shape[0]):
            means[i] = np.mean(trainset.images[:,i,:,:,:])
            stds[i] = np.std(trainset.images[:,i,:,:,:])
        #expand dimensions
        means = np.expand_dims(means, axis=-1)
        means = np.repeat(means, trainset.images.shape[-3], axis=-1)
        means = np.expand_dims(means, axis=-1)
        means = np.repeat(means,trainset.images.shape[-2], axis=-1)
        means = np.expand_dims(means, axis=-1)
        means = np.repeat(means,trainset.images.shape[-1], axis=-1)
        means = np.expand_dims(means, axis=0)


        stds = np.expand_dims(stds, axis=-1)
        stds = np.repeat(stds, trainset.images.shape[-3], axis=-1)
        stds = np.expand_dims(stds, axis=-1)
        stds = np.repeat(stds,trainset.images.shape[-2], axis=-1)
        stds = np.expand_dims(stds, axis=-1)
        stds = np.repeat(stds,trainset.images.shape[-1], axis=-1)
        stds = np.expand_dims(stds, axis=0)

        means_train = np.repeat(means, trainset.images.shape[0], axis=0)
        means_valid = np.repeat(means, validset.images.shape[0], axis=0)
        means_test = np.repeat(means, testset.images.shape[0], axis=0)


        stds_train = np.repeat(stds, trainset.images.shape[0], axis=0)
        stds_valid = np.repeat(stds, validset.images.shape[0], axis=0)
        stds_test = np.repeat(stds, testset.images.shape[0], axis=0)

        trainset.images = (trainset.images - means_train)/ stds_train
        validset.images = (validset.images - means_valid) / stds_valid
        testset.images = (testset.images - means_test) / stds_test

    elif input_dim == 4:
        for i in range(means.shape[0]):
            means[i] = np.mean(trainset.images[:,i,:,:])
            stds[i] = np.std(trainset.images[:,i,:,:])
        #expand dimensions
        means = np.expand_dims(means, axis=-1)
        means = np.repeat(means,trainset.images.shape[-2], axis=-1)
        means = np.expand_dims(means, axis=-1)
        means = np.repeat(means,trainset.images.shape[-1], axis=-1)
        means = np.expand_dims(means, axis=0)


        stds = np.expand_dims(stds, axis=-1)
        stds = np.repeat(stds,trainset.images.shape[-2], axis=-1)
        stds = np.expand_dims(stds, axis=-1)
        stds = np.repeat(stds,trainset.images.shape[-1], axis=-1)
        stds = np.expand_dims(stds, axis=0)

        means_train = np.repeat(means, trainset.images.shape[0], axis=0)
        means_valid = np.repeat(means, validset.images.shape[0], axis=0)
        means_test = np.repeat(means, testset.images.shape[0], axis=0)


        stds_train = np.repeat(stds, trainset.images.shape[0], axis=0)
        stds_valid = np.repeat(stds, validset.images.shape[0], axis=0)
        stds_test = np.repeat(stds, testset.images.shape[0], axis=0)

        trainset.images = (trainset.images - means_train)/ stds_train
        validset.images = (validset.images - means_valid) / stds_valid
        testset.images = (testset.images - means_test) / stds_test

    return  trainset, validset, testset


def _get_derivative_xy(trainset, validset, testset):
    for dataset in (trainset, validset, testset):
        blanche_x = dataset.images[:,:,1:2,:]
        blanche_y = dataset.images[:,:,:,1:2]
        drv_x = dataset.images[:,:,:,:] - np.concatenate((blanche_x, dataset.images[:,:,:-1,:]), axis=2)
        drv_x[:,:,0,:] = - drv_x[:,:,0,:]
        drv_y = dataset.images[:,:,:,:] - np.concatenate((blanche_y, dataset.images[:,:,:,:-1]), axis=3)
        drv_y[:,:,:,0] = - drv_y[:,:,:,0]
        dataset.images = np.concatenate((drv_x, drv_y), axis=1)

    return trainset, validset, testset


