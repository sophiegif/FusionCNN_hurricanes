## Module to read and process data from
## https://ghrc.nsstc.nasa.gov/services/storms website
import requests
from xml.etree import ElementTree
import re
import pickle
from http.client import RemoteDisconnected
import csv
import os
import pandas as pd
import numpy as np
from netCDF4 import num2date, date2num
from datetime import datetime, timedelta
from DataProcessing import ModuleReanalysisData as Mre


class Track:
    def __init__(self):
        self.dates = []
        self.categories=[]
        self.latitudes=[]
        self.longitudes=[]
        self.windspeeds=[]
        self.pressures=[]

        self.stormid = 0
        self.Ninstants=0
        self.month=None
        self.maxcategory=0 # out of 8 : 8 is maximum

    def import_from_raw_track(self, raw_track, stormid):
        self.stormid=stormid
        self.Ninstants=len(raw_track)
        for data_t in raw_track:
            date_t=list(map(int, re.findall('\d+',data_t['date'])))
            self.dates.append(date_t)
            self.categories.append(get_num_cat(data_t['category']))
            self.latitudes.append( float(data_t['latitude']) )
            self.longitudes.append(float(data_t['longitude']))
            self.pressures.append(float(data_t['pressure']))
            self.windspeeds.append(float(data_t['windspeed']))

        self.month=self.dates[0][1]
        self.maxcategory=max(self.categories)


    def import_from_raw_track_IBTRACKS(self,rootgrp, id_in_list, time_steps):

        self.stormid=b''.join(rootgrp['storm_sn'][id_in_list]).decode("utf-8")
        self.Ninstants=len(time_steps)

        for t in time_steps:
            dtime = num2date(rootgrp['time_wmo'][id_in_list][t], units=rootgrp['time_wmo'].units)
            self.dates.append([dtime.year,dtime.month,dtime.day,dtime.hour])
            wind=rootgrp['wind_wmo'][id_in_list][t]
            self.windspeeds.append(wind)
            self.categories.append(get_num_cat(sust_wind_to_cat(wind )) ) #1.12*

            self.latitudes.append(rootgrp['lat_wmo'][id_in_list][t])
            self.longitudes.append(rootgrp['lon_wmo'][id_in_list][t])
            self.pressures.append(rootgrp['pres_wmo'][id_in_list][t])

        self.month = self.dates[0][1]
        self.maxcategory = max(self.categories)
        if self.maxcategory>5: print('cat>5!!: '+str(self.maxcategory))


class Track_IBTRACKS_full(Track):
    def __init__(self):
        Track.__init__(self)
        self.name = ''
        self.basin=[]
        self.dist2land=[]
        self.nature=[] #Storm nature
                #  key: 0 = TS - Tropical
                #               1 = SS - Subtropical
                #               2 = ET - Extratropical
                #               3 = DS - Disturbance
                #               4 = MX - Mix of conflicting reports
                #               5 = NR - Not Reported
                #               6 = MM - Missing
                #               7 =  - Missing
         #(storm, time) Minimum Central Pressure
        # basin: Based on present location
        #  key: 0 = NA - North Atlantic
        # 1 = SA - South Atlantic
        # 2 = WP - West Pacific
        # 3 = EP - East Pacific
        # 4 = SP - South Pacific
        # 5 = NI - North Indian
        # 6 = SI - South Indian
        # 7 = AS - Arabian Sea
        # 8 = BB - Bay of Bengal
        # 9 = EA - Eastern Australia
        # 10 = WA - Western Australia
        # 11 = CP - Central Pacific
        # 12 = CS - Carribbean Sea
        # 13 = GM - Gulf of Mexico
        # 14 = MM - Missing

    def import_from_raw_track_IBTRACKS_full(self,rootgrp, id_in_list, time_steps):
        Track.import_from_raw_track_IBTRACKS(self,rootgrp,id_in_list, time_steps)
        self.name=b''.join(rootgrp['name'][id_in_list]).decode("utf-8")
        for t in time_steps:
            self.basin.append(rootgrp['basin'][id_in_list][t])
            self.dist2land.append(rootgrp['dist2land'][id_in_list][t])
            self.nature.append(rootgrp['nature_wmo'][id_in_list][t])


def get_num_cat(raw_category):
    '''

    :param raw_category: storm category
    :return: numerical category in [0,7]
    '''
    if raw_category in ['LP', 'WV', 'DB']:
        cat=0
    elif raw_category in ['SD', 'TD', 'ED']:
        cat=1
    elif raw_category in ['SS','TS','ES']:
        cat=2
    elif raw_category[0] is 'H':
        cat=int(raw_category[1])+2
    else:
        print('No category found. cat=-1')
        cat=-1
    return cat

def sust_wind_to_cat(wind):
    # maximum sustained wind in kt (knot)
    if wind<=33: cat='TD' # <=33
    elif wind<=63.:  cat='TS'
    elif wind <=82.: cat='H1'
    elif wind <=95.: cat='H2'
    elif wind <=112.: cat='H3'
    elif wind <=136.: cat='H4'
    else: cat='H5'

    return cat

def get_num_basin(basin):
    '''
    :param basin : string of the basin type 'AT', 'EP' or 'CP'
    :return: numerical basin {'AT':0, 'EP':1, 'CP':2}
    '''
    dict_basin={'AT':0, 'EP':1, 'CP':2}
    return dict_basin[basin]

def get_num_basin2(basin):
    '''
    :param basin : string of the basin type 'AT', 'EP' or 'CP'
    :return: numerical basin - same as IBtracks {'AT':0, 'EP':3, 'CP':12}
    '''
    dict_basin={'AT':0, 'EP':3, 'CP':12}
    return dict_basin[basin]

def get_disp_long_lat(stormid, t0, t, list_tracks=None, storm=None):
    if not storm:
        if not list_tracks:
            list_tracks=load_list_tracks_from_pkl()
        for track in list_tracks:
            if track.stormid == stormid:
                storm=track
                break
    if storm.Ninstants<t+1:
        return None
    lo=storm.longitudes[t]-storm.longitudes[t0]
    la=storm.latitudes[t]-storm.latitudes[t0]
    if lo>100:
        lo=lo-360
    if lo<-100:
        lo=lo+360

    return [lo,la]


def list_storm_request(date_init=None,date_end=None,basin=None,mincat=None,maxcat=None, flag_onlyids=False):
    '''
    get the list of the storms (with relevant infos) from the some date or loc infos
    :param date_init: 'yyyy-mm-dd'
    :param date_end: 'yyyy-mm-dd'
    :param basin: 'AT', 'EP' or 'CP' : atlantic (tot 1757), eastern pacific(tot 1018) or central pacific(tot 77)
    :param mincat:  L - Unknown, disturbance, wave, or low pressure
                    D - Tropical, subtropical, or extratropical depression
                    S - Tropical, subtropical, or extratropical storm
                    1 - Category-1 hurricane
                    2, 3, 4, 5 (idem)
    :param maxcat: idem
    :param flag_onlyids: if the output is only a list of storm ids, set to True. Default is False
    :return: dict: storms{stormid:stormparams} OR list of ids
    '''
    list_attr=locals()
    print(list_attr)

    string_tot='https://ghrc.nsstc.nasa.gov/services/storms/search.pl?'

    lut_args= dict(date_init='from', date_end='thru', basin='basin', mincat='mincategory', maxcat='maxcategory')

    for key,value in list_attr.items():
        if value and key in lut_args.keys():
            string_tot = string_tot+lut_args[key]+'='+str(value)+'&'
    string_tot = string_tot[:-1] # erase last '&'
    print(string_tot)
    try:
        r = requests.get(string_tot, headers={'Connection':'close'})
    except:
        print('Error in the request, maybe wrong parameters.')
        raise

    tree = ElementTree.fromstring(r.content)

    if flag_onlyids:
        storms = []
        if tree.getchildren()[0].tag=='Error':
            print('Warning: No storm found with these parameters!')
        else:
            print(str(len(tree.getchildren()) ) + ' storms found.')
            for child in tree.iter('Storm'):
                storms.append( int(child.attrib['stormid']) )
    else:
        storms = {}
        if tree.getchildren()[0].tag=='Error':
            print('Warning: No storm found with these parameters!')
        else:
            print(str(len(tree.getchildren()) ) + ' storms found.')
            for child in tree.iter('Storm'):
                storms[int(child.attrib['stormid'])] = child.attrib

    return storms


def get_storm_track(stormid=2017011):
    '''
    Get the track of the storm
    :param stormid: id of the storm
    :return: track= list of dicts, each element is one time point and its associated values
    '''
    while True:
        try:
            r = requests.get('https://ghrc.nsstc.nasa.gov/services/storms/track.pl?stormid='+str(stormid),
                             headers={'Connection':'close'})
            tree = ElementTree.fromstring(r.content)
            if tree.getchildren()[0].tag == 'Error':
                print('Warning: No track found with this storm id!')
                raise IOError
            raw_track = []
            for child in tree.iter('Track'):
                raw_track.append(child.attrib)
            break
        except (RemoteDisconnected , ConnectionError):
            print(str(stormid)+ ' connection error! trying again...')
        except:
            print('Error in the track request, maybe wrong parameters:')
            print('current request is: '+'https://ghrc.nsstc.nasa.gov/services/storms/track.pl?stormid='+str(stormid))
            raw_track=[]

    return raw_track


def get_6h_step_storm(times_storm):
    times_storm=times_storm.compressed()
    time0=times_storm[0]
    store_times=[0]
    for time1,id_t in zip(times_storm[1:],range(1,len(times_storm))):
        h_diff=24*(time1-time0)
        if h_diff==6:
            time0=time1
            store_times.append(id_t)
        elif h_diff<6:
            pass
        elif h_diff>6:
            return 0
    return store_times


def save_all_storm_tracks_IBTRACKS(namefiledata, namefilesaving, date_init=1860, flag_full_data=False):

    rootgrp = Mre.open_netCDF_file(namefiledata)
    list_tracks = []
    for s in range(len(rootgrp['storm_sn'])):
        dtime = num2date(rootgrp['time_wmo'][s][0], units=rootgrp['time_wmo'].units)
        if dtime.year < date_init:
            continue

        time_steps=get_6h_step_storm(rootgrp['time_wmo'][s])
        if time_steps==0:
            continue

        if not flag_full_data:
            track=Track()
            track.import_from_raw_track_IBTRACKS(rootgrp,s,time_steps)
        else:
            track=Track_IBTRACKS_full()
            track.import_from_raw_track_IBTRACKS_full(rootgrp,s,time_steps)
        list_tracks.append(track)

    with open(namefilesaving, 'wb') as pickle_file:
        pickle.dump(list_tracks, pickle_file)


def load_list_tracks_from_pkl(pkl_inputfile):

    file=open(pkl_inputfile, 'rb')
    list_tracks=pickle.load(file)
    file.close()
    return list_tracks


def get_all_tracks_from_period(year_init=1958, year_end=2017):
    list_tracks=load_list_tracks_from_pkl()
    tracks=[]
    for track in list_tracks:
        if track.dates[0][0] in range(year_init, year_end+1):
            tracks.append(track)
    return tracks


def write_csv_data_from_pkl(pkl_inputfile, csv_outputfile, fields, windowt=8, thresh=4, flag_increase_data=True, namefilebasin=None):

    file=open(pkl_inputfile, 'rb')
    list_tracks=pickle.load(file)
    file.close()

    name_cols=['stormid','delay_tsteps', 'intenseStorm']
    for field in fields:
        if field in ['month', 'maxcategory']:
            name_cols.append(field)
        elif field is 'basin' and os.path.isfile(namefilebasin):
            name_cols.append(field)
            basins=dict(np.array(pd.read_csv(namefilebasin,header=None)))
        else:
            name_cols.extend([field+str(i) for i in range(windowt)])

    with open(csv_outputfile, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(name_cols)

        smalltracks=0
        maxcat_tracks=0
        Nintense=0
        for track in list_tracks:
            if track.Ninstants<windowt+2:
                print('Track '+str(track.stormid)+' too small, only '+str(track.Ninstants)+' instants.')
                smalltracks=smalltracks+1
                continue
            if thresh < max(track.categories[0:windowt])+1:
                print('Track '+str(track.stormid)+' threshold category ('+str(thresh)+') already reached.')
                maxcat_tracks=maxcat_tracks+1
                continue

            # augment size of database by sliding the initial window.
            if flag_increase_data:
                range_windows=range(min(int((track.Ninstants-windowt)/2),6))
            else:
                range_windows=[0]
            for delay in range_windows:
                if thresh < max(track.categories[0:windowt+delay*2])+1:
                    continue
                else:
                    tarray=[track.stormid, delay*2]
                    if track.maxcategory>thresh:
                        tarray.append(1)
                        Nintense=Nintense+1
                    else:
                        tarray.append(0)

                    for field in fields:
                        if field in ['month', 'maxcategory']:
                            tarray.append(getattr(track,field))
                        elif field is 'basin' and os.path.isfile(namefilebasin):
                            tarray.append(get_num_basin(basins[track.stormid]))
                        else:
                            tarray.extend(getattr(track,field)[delay*2:windowt+delay*2])
                    spamwriter.writerow(tarray)
        print('Number of too small tracks: '+str(smalltracks))
        print('Number of threshold category already reached: '+ str(maxcat_tracks))
        print('Number of intense (positive) storms: '+str(Nintense))


def get_sliding_tracks_from_pkl(windowi=8, instant_f=8, year_init=1958, year_end=2017, flag_windspeed=True, flag_abs_degrees=False):
    '''
    get X and y for learning storm direction from only tracking data.
    :param windowi: nb of time points to use in training
    :param instant_f: time point target
    :param year_init:
    :param year_end:
    :param flag_windspeed: if windspeeds in train
    :return: X, y, group
        X=matrix of features(nb_samplesxnb_features)
        y= matrix of target, delta long/lat at instant_f (nb_samplesx2),
        group=list of id of the storm, for chosing train/test data)
    '''
    print('ModuleStormReader.get_sliding_tracks_from_pkl function...')
    list_tracks=get_all_tracks_from_period(year_init, year_end)

    X=[]; y=[]; group=[]
    Nsmalltracks = 0

    for track in list_tracks:

        if track.Ninstants<instant_f+1:
            Nsmalltracks=Nsmalltracks+1
            continue
        # augment size of database by sliding the initial window. (here delay = 1 time step)
        range_windows = range(track.Ninstants - instant_f)
        for delay in range_windows:
            x_i=[]
            for i in range(delay, delay+windowi-1):
                if flag_windspeed:
                    x_i.append(track.windspeeds[i])
                if flag_abs_degrees:
                    x_i.append(track.longitudes[i])
                    x_i.append(track.latitudes[i])
                x_i.extend( get_disp_long_lat(track.stormid,i,i+1, storm=track) )
            if flag_windspeed:
                x_i.append(track.windspeeds[delay+windowi-1])
            if flag_abs_degrees:
                x_i.append(track.longitudes[delay+windowi-1])
                x_i.append(track.latitudes[delay+windowi-1])
            y_i=get_disp_long_lat(track.stormid,windowi-1+delay,instant_f+delay, storm=track)

            X.append(x_i)
            y.append(y_i)
            group.append(track.stormid)

    print('X and y created.')
    print('  Nb of samples:'+ str(len(X)))
    print('  Nb of storms used: '+str(len(list_tracks)))
    print('  Nb of too small tracks: '+str(Nsmalltracks))

    return X,y,group


def load_1D_data_from_IBTRACS(list_stormids,names_1Ddata,
                              file_IBtracks='/home/sgiffard/Documents/StormProject/DataStorm/storm_IBTrACS/tracks_1979_after_full_data_cat_nature.pkl',
                              file_tracksold='/home/sgiffard/Documents/StormProject/DataStorm/2018_02_04_processed_pickle/tracks_1860-01-01_after.pkl',
                              filecorr_stormids='/home/sgiffard/Documents/StormProject/DataStorm/storm_IBTrACS/correspondances_stormids.txt',
                              namefilebasin='/home/sgiffard/Documents/StormProject/DataStorm/2018_02_04_processed_pickle/basins_idstorms.csv'):
    with open(filecorr_stormids, 'r') as f:
        LUT_to_IBTRACS = {}
        for line in f:
            p = line.split()
            LUT_to_IBTRACS[p[0]] = p[1]

    list_IBtracks= load_list_tracks_from_pkl(file_IBtracks)
    dict_IBtracks={}
    for track in list_IBtracks:
        dict_IBtracks[track.stormid]=track
    list_oldtracks = load_list_tracks_from_pkl(file_tracksold)
    dict_oldtracks={}
    for track in list_oldtracks:
        dict_oldtracks[str(track.stormid)]=track

    basins_old=dict(np.array(pd.read_csv(namefilebasin,header=None)))

    list_1Ddatatot=[]
    for id in list_stormids:
        flag_old=False

        if str(id) in LUT_to_IBTRACS.keys():
            oldid=LUT_to_IBTRACS[str(id)]
        else:
            oldid=None
        if str(id) not in dict_IBtracks.keys():
            if str(id) in LUT_to_IBTRACS.keys():
                newid=LUT_to_IBTRACS[str(id)]
            else:
                flag_old=True
                newid=id
        else:  newid=id

        if flag_old: track=dict_oldtracks[str(newid)]
        else:
            track=dict_IBtracks[str(newid)]
            if oldid:
                trackold = dict_oldtracks[oldid]

        for t in range(track.Ninstants):
            if oldid:
                t_f = None
                for t2 in range(trackold.Ninstants):
                    if track.dates[t] == trackold.dates[t2]:
                        t_f=t2
                if not t_f:
                    continue
            else:
                t_f=t
            list_1Ddata=[id, t_f]
            for name in names_1Ddata:
                if name =='windspeed':
                    list_1Ddata.append(track.windspeeds[t])
                elif name =='Jday_predictor': # static (at t=0)
                    dtime = datetime(track.dates[0][0], track.dates[0][1], track.dates[0][2], track.dates[0][3])
                    Jnum=date2num(dtime, "days since 1900-01-01", "standard")
                    if str(id).find('S')>0: # pacific
                        season_peak=date2num(datetime(track.dates[0][0],1,1),"days since 1900-01-01",
                                             "standard")+238
                    else: # atlantic
                        season_peak = date2num(datetime(track.dates[0][0], 1, 1), "days since 1900-01-01",
                                               "standard") + 253
                    Rd=25 # days providing the best fit, according to demaria 2005 p. 535.
                    list_1Ddata.append(np.exp(-np.power((Jnum-season_peak)/Rd,2)))
                elif name == 'hemisphere':
                    if str(id).find('S') > 0:  # pacific
                        list_1Ddata.append(0)
                    else:
                        list_1Ddata.append(1)
                elif name =='latitude':
                    list_1Ddata.append(track.latitudes[t])
                elif name =='longitude':
                    list_1Ddata.append(track.longitudes[t])
                elif name =='initial_max_wind': #static
                    list_1Ddata.append(track.windspeeds[0])
                elif name =='max_wind_change_12h':
                    if t==0:
                        list_1Ddata.append(0)
                    elif t==1:
                        list_1Ddata.append(track.windspeeds[t] - track.windspeeds[t-1])
                    else:
                        list_1Ddata.append(track.windspeeds[t]-track.windspeeds[t-2])
                elif name=='basin':
                    if flag_old:
                        list_1Ddata.append(get_num_basin2(basins_old[track.stormid]))

                    else:    list_1Ddata.append(track.basin[t])
                elif name =='dist2land':
                    if flag_old:
                        list_1Ddata.append(None)
                    else:   list_1Ddata.append(track.dist2land[t])
                elif name =='nature':
                    if flag_old:
                        list_1Ddata.append(None)
                    else:   list_1Ddata.append(track.nature[t])
            list_1Ddatatot.append(list_1Ddata)

    return list_1Ddatatot


def load_1D_data_from_IBTRACS_simple(names_1Ddata,
                              file_IBtracks='/home/sgiffard/Documents/StormProject/DataStorm/storm_IBTrACS/tracks_1979_after_full_data_cat_nature.pkl'):

    list_IBtracks= load_list_tracks_from_pkl(file_IBtracks)
    list_1Ddatatot=[]

    for track in list_IBtracks:
        for t in range(track.Ninstants):

            list_1Ddata=[track.stormid, t]
            for name in names_1Ddata:
                if name =='windspeed':
                    list_1Ddata.append(track.windspeeds[t])
                elif name =='Jday_predictor': # static (at t=0)
                    dtime = datetime(track.dates[0][0], track.dates[0][1], track.dates[0][2], track.dates[0][3])
                    Jnum=date2num(dtime, "days since 1900-01-01", "standard")
                    if str(track.stormid).find('S')>0: # pacific
                        season_peak=date2num(datetime(track.dates[0][0],1,1),"days since 1900-01-01",
                                             "standard")+238
                    else: # atlantic
                        season_peak = date2num(datetime(track.dates[0][0], 1, 1), "days since 1900-01-01",
                                               "standard") + 253
                    Rd=25 # days providing the best fit, according to demaria 2005 p. 535.
                    list_1Ddata.append(np.exp(-np.power((Jnum-season_peak)/Rd,2)))
                elif name == 'hemisphere':
                    if str(track.stormid).find('S') > 0:  # pacific
                        list_1Ddata.append(0)
                    else:
                        list_1Ddata.append(1)
                elif name =='latitude':
                    list_1Ddata.append(track.latitudes[t])
                elif name =='longitude':
                    list_1Ddata.append(track.longitudes[t])
                elif name =='initial_max_wind': #static
                    list_1Ddata.append(track.windspeeds[0])
                elif name =='max_wind_change_12h':
                    if t==0:
                        list_1Ddata.append(0)
                    elif t==1:
                        list_1Ddata.append(track.windspeeds[t] - track.windspeeds[t-1])
                    else:
                        list_1Ddata.append(track.windspeeds[t]-track.windspeeds[t-2])
                elif name=='basin':
                        list_1Ddata.append(track.basin[t])
                elif name =='dist2land':
                        list_1Ddata.append(track.dist2land[t])
                elif name =='nature':
                        list_1Ddata.append(track.nature[t])
            list_1Ddatatot.append(list_1Ddata)

    return list_1Ddatatot
