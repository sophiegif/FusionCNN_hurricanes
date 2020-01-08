# Copyright (c) 2020 Sophie Giffard-Roisin <sophie.giffard@univ-grenoble-alpes.fr>
# SPDX-License-Identifier: GPL-3.0

from __future__ import print_function
import sys
import numpy as np
from collections import Counter
from Utils.MyDataset import MyDataset
import pickle
from tqdm import tqdm
import pandas as pd

X_DIR = '/data/Xy/2018_04_10_ERA_interim_storm/X_pl_crop25_z_u_v/'
X_DIR2 = '/data/Xy/2018_04_10_ERA_interim_storm/X_pl_crop25_z_u_v_historic6h/'
Y_DIR_D = '/data/Xy/2018_04_10_ERA_interim_storm/y_disp2/'
Data_stormids_csv = "/data/Xy/2018_04_10_ERA_interim_storm/1D_data_matrix_IBTRACS.csv"



def load_datasets(sample=0.1, list_params=None, load_t = (0, -6), valid_split=0.2, test_split=0.2, randomseed=47):
    """
    load X1, Y, storm ids from raw pkl files
    :param list_params:
    :param sample: the proportion of sampling, Set sample=1.0 to get the whole data
    :param category: type of label to load, if True, load category as label, if False, load deplacement
    :param localtest: If True, load data from local directory, if False, load from cluster
    :return: X1,Y,storm_ids
    """

    if list_params is None:
        list_params = ['r', 'd', 'o3', 'v', 'ciwc', 'q', 'pv', 'z', 'clwc', 't', 'w', 'vo', 'u', 'cc']
    X1 = []
    X2 = []
    Y = []
    storm_ids = []
    print('loading from pkls ...')
    # load which y
    x_dir = X_DIR
    x_dir2 = X_DIR2
    y_dir = Y_DIR_D

    data = pd.read_csv(Data_stormids_csv)
    stormids = np.unique(data['stormid'].values)
    for filename in tqdm(stormids):
        non_empty_storm = False  # check if the storm is empty
        if np.random.random() > sample:
            continue
        else:
            if 0 in load_t:
                with open(x_dir+filename+'.pkl', 'rb') as f:
                    storm_data = pickle.load(f)
                if len(storm_data['grids']) != 0:
                    non_empty_storm = True
                    storm_i = []
                    for storm_t in storm_data['grids'][:]:
                        grid_allchannels = []
                        for key in list_params:
                            grid = storm_t[key]
                            grid_allchannels.append(grid)
                        storm_i.append(grid_allchannels)
                    X1.append(storm_i)
                    storm_ids.append(filename)
                    del storm_data
            if -6 in load_t:
                with open(x_dir2+filename+'.pkl', 'rb') as f:
                    storm_data = pickle.load(f)
                if len(storm_data['grids']) != 0:
                    non_empty_storm = True
                    storm2_i = []
                    for storm_t in storm_data['grids'][:]:
                        grid_allchannels = []
                        for key in list_params:
                            grid = storm_t[key]
                            grid_allchannels.append(grid)
                        storm2_i.append(grid_allchannels)
                    X2.append(storm2_i)
                    del storm_data
            for item in load_t:
                if item not in (0,-6,-12):
                    raise ValueError('only support for t, t-6, t-12')

            with open(y_dir+filename+'.pkl', 'rb') as f:
                y_data = pickle.load(f)
            if non_empty_storm is True:
                y_labels = []
                n_storms = len(y_data['next_disp'])
                for i in range(n_storms):
                    y_label_i = [y_data['curr_longlat'][i], y_data['curr_longlat'][i+1], y_data['next_disp'][i]]
                    y_labels.append(y_label_i)
                    Y.append(y_labels)

    # set indices for train, valid, test
    num_storm = len(X1)
    indices = list(range(num_storm))
    num_test = int(np.floor(test_split * num_storm))
    num_valid = int(np.floor(valid_split * num_storm))
    np.random.seed(randomseed)
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx =\
        indices[:-num_valid-num_test], indices[-num_valid-num_test:-num_test], indices[-num_test:]

    trainset = load_foldset(train_idx, storm_ids, Y, X1, X2)
    validset = load_foldset(valid_idx, storm_ids, Y, X1, X2)
    testset = load_foldset(test_idx, storm_ids, Y, X1, X2)

    return trainset, validset, testset



def load_foldset(fold_idx, storm_ids, Y, X1, X2=None):
    """
    :return: foldset (train, test ou valid)
    """
    first_non_empty_X=True
    for X in (X1, X2):
        if X != [] and X is not None:
            if first_non_empty_X is True:
                # !!called train in the future, but it actually depend on the fold.

                fold_X = list(X[i] for i in fold_idx)
                fold_Y = list(Y[i] for i in fold_idx)
                fold_ids = list(storm_ids[i] for i in fold_idx)

                # put datapoints from different storm to a big list
                for idx, storm in enumerate(fold_X):
                    fold_ids[idx] = np.repeat(fold_ids[idx],len(storm))
                    fold_ids = reduce_dim(fold_ids)
                fold_X = reduce_dim_float32(fold_X)
                fold_Y = reduce_dim_float32(fold_Y)

                #set Y to be double float
                fold_Y = np.double(fold_Y)

                # convert nan to 0
                np.nan_to_num(fold_X, copy=False)
                np.nan_to_num(fold_Y, copy=False)

                # get list of timesteps
                fold_timestep = []
                fold_indexes = np.unique(fold_ids, return_index=True)[1]
                fold_ids_list = [fold_ids[index] for index in sorted(fold_indexes)]
                fold_ids_count = Counter(fold_ids)
                for id in fold_ids_list:
                    fold_timestep.extend(list(range(fold_ids_count[id])))

                # set flag 'first_non_empty_X' to be False
                first_non_empty_X = False

            else:
                fold_X2 = list(X[i] for i in fold_idx)
                fold_X2 = reduce_dim_float32(fold_X2)
                fold_X = np.concatenate((fold_X, fold_X2), axis=1)

    # foldset can be either train, test ou valid.
    foldset = MyDataset(fold_X, fold_Y, fold_ids, fold_timestep)

    return foldset


def reduce_dim(X):
    X_reduced = []
    for x in tqdm(X):
        X_reduced.extend(x)
    X_reduced_array = np.array(X_reduced)
    return X_reduced_array


def reduce_dim_float32(X):
    X_reduced = []
    for x in tqdm(X):
        X_reduced.extend(x)
    X_reduced_array = np.array(X_reduced, dtype=np.float32)
    return X_reduced_array


