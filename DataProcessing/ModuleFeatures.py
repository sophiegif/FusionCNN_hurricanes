import pickle

from DataProcessing import ModuleStormReader as Msr
from DataProcessing import ModuleReanalysisData as Mre
from DataProcessing import MyModuleFileFolder as MMff


def load_data_storm_interim(types, size_crop=7, levtype='sfc', pkl_inputfile='./data/tracks.pkl', folder_data='./data/',
                             folder_saving='./features/', folderLUT='./data/', levels=None, flag_write_ys=True,
                             flag_write_X=True, history=0):

    print(levels, flush=True)
    print(flag_write_ys, flush=True)
    print(flag_write_X, flush=True)
    print(folder_saving, flush=True)
    print(levtype, flush=True)
    print(history, flush=True)
    list_tracks=Msr.load_list_tracks_from_pkl(pkl_inputfile=pkl_inputfile)
    dict_tracks={}
    for track in list_tracks:
        dict_tracks[track.stormid]=track

    year_curr=list_tracks[0].dates[0][0]

    for track in list_tracks:
        # if track.dates[0][0]< 2010: # or track.dates[0][0]>1988:
        #     continue

        year = track.dates[0][0]
        if year>year_curr:
            print(str(year), flush=True)
            year_curr=year
        range_windows = range(int(track.Ninstants-1))

        ### get X ###
        # get grid data
        if flag_write_X:
            X=Mre.get_windows_from_track_interim(track, range_windows, types, size_crop=size_crop,
                                       folder_data=folder_data, levtype=levtype, folderLUT=folderLUT,
                                                 levels=levels,history=history)
        if flag_write_ys:
            y_curr_cat = []
            y_next_disp = []
            y_curr_longlat=[]
            y_windspeed=[]
            for delay in range_windows:
                #### get Y #######
                y_curr_cat.append(track.categories[delay])
                y_next_disp.append(Msr.get_disp_long_lat(track.stormid, delay,
                                          delay +1,storm=dict_tracks[track.stormid]))
                y_curr_longlat.append([track.longitudes[delay],track.latitudes[delay]])
                y_windspeed.append(track.windspeeds[delay])
            y_curr_cat.append(track.categories[-1])
            y_curr_longlat.append([track.longitudes[-1], track.latitudes[-1]])
            y_windspeed.append(track.windspeeds[-1])

        #### write files #####
        if flag_write_X:
            folder_saving_X=folder_saving+'X_'+levtype+'_crop'+str(size_crop)+'_r_vo_w/'
            if history:
                folder_saving_X=folder_saving_X[:-1]+'_historic'+str(6*history)+'h/'
            print(folder_saving_X, flush=True)
            MMff.MakeDir(folder_saving_X)
            with open(folder_saving_X+str(track.stormid)+'.pkl', 'wb') as file_X:
                pickle.dump({'grids':X},file_X)
        if flag_write_ys:
            MMff.MakeDir(folder_saving + 'y_categories2/')
            infos='curr_cat : current category of the storm (between 0 and 7). From t=0 to t=n \n'
            with open(folder_saving + 'y_categories2/' + str(track.stormid) + '.pkl', 'wb') as file_cats:
                pickle.dump({'curr_cat': y_curr_cat, 'infos':infos}, file_cats)
            MMff.MakeDir(folder_saving + 'y_disp2/')
            infos = 'next_disp : next displacement (delta(longitude),delta(latitude)), in degree. From t=0 to t=n-1 \n'
            infos = infos+'curr_longlat: current longitude and latitude in degrees. From t=0 to t=n'
            with open(folder_saving + 'y_disp2/' + str(track.stormid) + '.pkl', 'wb') as file_disp :
                pickle.dump({'next_disp': y_next_disp, 'curr_longlat':y_curr_longlat, 'infos':infos}, file_disp)
            MMff.MakeDir(folder_saving + 'y_wind2/')
            infos = 'Current mean value of maximum sustained winds using a 10-minute average. \nIn knots. \nFrom t=0 to t=1.'
            with open(folder_saving + 'y_wind2/' + str(track.stormid) + '.pkl', 'wb') as file_wind :
                pickle.dump({'curr_wind': y_windspeed, 'infos': infos}, file_wind)

    print('Extracting data complete!', flush=True)

