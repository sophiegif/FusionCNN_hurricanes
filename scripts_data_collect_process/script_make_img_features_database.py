import DataProcessing.ModuleReanalysisData as Mre
import DataProcessing.ModuleFeatures as Ml

folder_data_tot = './data/'
folderLUT = folder_data_tot
foldersaving=folder_data_tot+'Xy/'
pkl_inputfile = folder_data_tot+'tracks_IBTRACKS_1979_after.pkl'

size_grid = 1
size_crop = 11
levtype = 'pl' # or 'sfc'
flag_write_ys = False # write the target (y)
flag_write_X = True # write the input data (X)

possible_levels = Mre.open_level_list('pl', folderLUT=folderLUT)
levelist = [lev for lev in possible_levels if int(lev)>90] # no level smaller than 100 hPa
levelist = levelist[1::2]

LUTfilename=folderLUT+'list_params_nums_'+levtype+'.txt'
shortnames, ids, names, namesCDF=Mre.open_LUT_list_params(LUTfilename)
total_list_params=shortnames

list_params = ['r', 'vo','w', 'z','u','v']
history = 0 # or 1, or 2 (number of historical time steps to also store (1= 6h behind stored, 2=6h and 2h behind stored)
folderpath=folder_data_tot+'ERA_interim/'\
           +'grid_'+str(size_grid)+'/'+levtype+'/'+'_'.join(total_list_params)+'/'

# check if params in list params:
if not set(list_params).issubset(shortnames):
    print('Warning! list_params is not a subset of possible parameters!')

#check if levelist in possible levels:
if levelist:
    possible_levels = Mre.open_level_list('pl', folderLUT=folderLUT)
    if not set(levelist).issubset(possible_levels):
        print('Warning! levelist is not a subset of possible levels!')

Ml.load_data_storm_interim( list_params, size_crop=size_crop, levtype=levtype, pkl_inputfile=pkl_inputfile,
                            folder_data=folderpath, folder_saving=foldersaving, folderLUT=folderLUT,
                            levels=levelist, flag_write_ys=flag_write_ys, flag_write_X=flag_write_X,
                            history=history)