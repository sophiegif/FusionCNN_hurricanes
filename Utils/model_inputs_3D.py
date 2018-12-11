from __future__ import print_function
import sys
sys.path.append('/home/tau/myang/ClimateSaclayRepo/')
import os
import numpy as np
from collections import Counter
from Utils.MyDataset import MyDataset
import pickle
from tqdm import tqdm
import pandas as pd

X_DIR='/data/titanic_1/users/sophia/sgiffard/data/Xy/2018_04_10_ERA_interim_storm/X_pl_crop25_z_u_v/'
X_DIR2='/data/titanic_1/users/sophia/sgiffard/data/Xy/2018_04_10_ERA_interim_storm/X_pl_crop25_z_u_v_historic6h/'
X_DIR3 = '/data/titanic_1/users/sophia/sgiffard/data/Xy/2018_04_10_ERA_interim_storm/X_pl_crop25_z_u_v_historic12h/'
Y_DIR_C='/data/titanic_1/users/sophia/sgiffard/data/Xy/2018_04_10_ERA_interim_storm/y_categories2/'
Y_DIR_D='/data/titanic_1/users/sophia/sgiffard/data/Xy/2018_04_10_ERA_interim_storm/y_disp2/'


def load_datasets(sample = 0.1, category=False, list_params=None,load_t = (0, -6, -12), valid_split=0.2, test_split=0.2, randomseed=47):
    """
    load X1, Y, storm ids from raw pkl files
    :param list_params:
    :param sample: the propostion of sampling, Set sample=1.0 to get the whole data
    :param category: type of label to load, if True, load category as label, if False, load deplacement
    :param localtest: If True, load data from local directory, if False, load from cluster
    :return: X1,Y,storm_ids
    """

    if list_params is None:
        list_params = ['r', 'd', 'o3', 'v', 'ciwc', 'q', 'pv', 'z', 'clwc', 't', 'w', 'vo', 'u', 'cc']
    X1 = []
    X2 = []
    X3 = []
    Y = []
    storm_ids = []
    print('loading from pkls ...')
    # load which y
    x_dir = X_DIR
    x_dir2 = X_DIR2
    x_dir3 = X_DIR3
    if category == True:
        y_dir = Y_DIR_C
    else:
        y_dir = Y_DIR_D

    data = pd.read_csv("/data/titanic_1/users/sophia/sgiffard/data/Xy/2018_04_10_ERA_interim_storm/1D_data_matrix_IBTRACS.csv")
    stormids = np.unique(data['stormid'].values)
    for filename in tqdm(stormids):
        non_empty_storm = False #check if the storm is empty
        if np.random.random()>sample:
            continue
        else:
            if 0 in load_t:
                with open(x_dir+filename+'.pkl', 'rb') as f:
                    storm_data = pickle.load(f)
                if len(storm_data['grids'])!=0:
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
                if len(storm_data['grids'])!=0:
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
            if -12 in load_t:
                with open(x_dir3+filename+'.pkl', 'rb') as f:
                    storm_data = pickle.load(f)
                if len(storm_data['grids'])!=0:
                    non_empty_storm=True
                    storm3_i = []
                    for storm_t in storm_data['grids'][:]:
                        grid_allchannels = []
                        for key in list_params:
                            grid = storm_t[key]
                            grid_allchannels.append(grid)
                        storm3_i.append(grid_allchannels)
                    X3.append(storm3_i)
                    del storm_data
            for item in load_t:
                if item not in (0,-6,-12):
                    raise ValueError('only support for t, t-6, t-12')

            with open(y_dir+filename+'.pkl', 'rb') as f:
                y_data = pickle.load(f)
            if non_empty_storm == True:
                if category == True:
                    Y.append(y_data['curr_cat'][1:])
                else:
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

    trainset = load_foldset(train_idx, storm_ids, Y, X1, X2, X3)
    validset = load_foldset(valid_idx, storm_ids, Y, X1, X2, X3)
    testset = load_foldset(test_idx, storm_ids, Y, X1, X2, X3)

    return trainset, validset, testset



def load_foldset(fold_idx, storm_ids, Y, X1, X2=None, X3=None):
    """
    :return: foldset (train, test ou valid)
    """
    first_non_empty_X=True
    for X in (X1, X2, X3):
        if X != [] and X is not None:
            if first_non_empty_X == True:

                # !!called train in the future, but it actually depend on the fold.

                train_X = list(X[i] for i in fold_idx)
                train_Y = list(Y[i] for i in fold_idx)
                train_ids = list(storm_ids[i] for i in fold_idx)

                # put datapoints from different storm to a big list
                for idx, storm in enumerate(train_X):
                    train_ids[idx] = np.repeat(train_ids[idx],len(storm))
                train_ids = reduce_dim(train_ids)
                train_X = reduce_dim_float32(train_X)
                train_Y = reduce_dim_float32(train_Y)

                #set Y to be double float
                train_Y = np.double(train_Y)

                # convert nan to 0
                np.nan_to_num(train_X,copy=False)
                np.nan_to_num(train_Y,copy=False)

                # get list of timesteps
                train_timestep = []
                train_indexes = np.unique(train_ids, return_index=True)[1]
                train_ids_list = [train_ids[index] for index in sorted(train_indexes)]
                train_ids_count = Counter(train_ids)
                for id in train_ids_list:
                    train_timestep.extend(list(range(train_ids_count[id])))

                # set flag 'first_non_empty_X' to be False
                first_non_empty_X = False

            else:
                train_X2 = list(X[i] for i in fold_idx)
                train_X2 = reduce_dim_float32(train_X2)
                X=[]
                train_X = np.concatenate((train_X, train_X2), axis=1)

    # foldset can be either train, test ou valid.
    foldset = MyDataset(train_X,train_Y, train_ids, train_timestep)

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
    X_reduced_array = np.array(X_reduced,dtype=np.float32)
    return X_reduced_array


