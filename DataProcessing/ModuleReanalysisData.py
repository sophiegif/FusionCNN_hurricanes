# Copyright (c) 2020 Sophie Giffard-Roisin <sophie.giffard@univ-grenoble-alpes.fr>
# SPDX-License-Identifier: GPL-3.0

from ftplib import FTP
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import calendar

from ecmwfapi import ECMWFDataServer
from DataProcessing import ModuleStormReader as Msr
from DataProcessing import MyModuleFileFolder as MMff


def list_to_datetime(list):
    if len(list)==5:
        date_t=datetime(list[0],list[1],list[2],list[3], list[4])
    elif len(list)==4:
        date_t=datetime(list[0],list[1],list[2],list[3])
    else:
        date_t=datetime(list[0],list[1],list[2])
    return date_t


def open_netCDF_file(nc_filename):
    rootgrp = Dataset(nc_filename, "r+", format="NETCDF4")
    return rootgrp


def get_cropped_nefCDF_data_interim(rootgrp, params_netCDF=['u10'],
                            _datetime=datetime(2000,1,1,6), center=[-95,25], size=20,
                            params_shortnames=None, levels=None, flag_save_lonlat=False):
    '''
    Get cropped data around a center from ERA interim data, several parameters can be acquired at the same time.
    :param rootgrp: netCDF file corresponding to the date wanted.
    :param params_netCDF: names of the parameters to get, as they appear in the netCDF (rootgrp) file
    :param _datetime:
    :param center: center of the grid (location of the storm)
    :param size: spatial length of the window: number of points (and not number of degrees)
    :return: grid = dict(param_netCDF : sizexsize), longs, lats
    '''
    # correct shortnames (the netCDF names are sometimes only numbers...)
    if not params_shortnames or len(params_shortnames) != len(params_netCDF) :
        params_shortnames=params_netCDF

    nlats = len(rootgrp.dimensions["latitude"])
    nlons = len(rootgrp.dimensions["longitude"])
    size_grid_lat=rootgrp['latitude'][1]-rootgrp['latitude'][0]
    size_grid_lon=rootgrp['longitude'][1]-rootgrp['longitude'][0]

    # range longitudes (indices):
    if center[0] < 0:
        center[0] = center[0]+360
    # size points and not size in degrees
    approx_center0=round(center[0]/size_grid_lon)
    min_lon=approx_center0-int((size-1)/2)
    max_lon=approx_center0+int((size-1)/2)
    range_ilon = list(map(int, range(int(min_lon),int(max_lon+1) )) )
    range_ilon_new=[]
    for ilon in range_ilon:
        if ilon<0:  range_ilon_new.append(ilon+nlons)
        elif ilon>=nlons:    range_ilon_new.append(ilon-nlons)
        else:   range_ilon_new.append(ilon)

    # range latitudes (indices):
    center_i1=round((center[1]+90)/size_grid_lat)
    min_lat=center_i1-int((size-1)/2)
    max_lat=center_i1+int((size-1)/2)
    range_ilat=list(map(int, range(int(min_lat)-1, int(max_lat) ))) # after it will be in reverse!
    range_ilat_new=[]
    for ilat in range_ilat:
        if ilat<0:    range_ilat_new.append(ilat+nlats)
        elif ilat>nlats:    range_ilat_new.append(ilat-nlats)
        else:  range_ilat_new.append(ilat)

    # date (in hours since ...depend! --> one value every 6 hours)
    times = rootgrp['time']
    time_i = date2num(_datetime, units=times.units)-times[0]
    time_i=int(time_i/6)


    # final grid
    grid={}; longs=[]; lats = []
    for param,param_name in zip(params_netCDF,params_shortnames):
        grid[param_name]=[]
        if levels==None:
            for i_lat, i in zip(reversed(range_ilat_new), range(len(range_ilon_new))):  # reversed because latitudes are filled from +90 to -90!
                grid[param_name].append([])
                if flag_save_lonlat:
                    lats.append(rootgrp['latitude'][i_lat])
                    longs = []
                for i_lon in range_ilon_new:
                    grid[param_name][i].append(rootgrp[param][time_i][i_lat][i_lon])
                    if flag_save_lonlat:
                        longs.append(rootgrp['longitude'][i_lon])
        else:
            i_levs=[]
            for level,i_lev in zip(rootgrp['level'],range(len(rootgrp['level']))):
                i_levs.append(i_lev)

            for i_lev,i_lev_final in zip(i_levs,range(len(i_levs))):
                grid[param_name].append([])
                for i_lat, i in zip(reversed(range_ilat_new), range(len(range_ilon_new))):  # reversed because latitudes are filled from +90 to -90!
                    if flag_save_lonlat:
                        lats.append(rootgrp['latitude'][i_lat])
                    grid[param_name][i_lev_final].append(rootgrp[param][time_i][i_lev][i_lat][range_ilon_new])
                    if flag_save_lonlat:
                        longs= rootgrp['longitude'][range_ilon_new]


    return grid,longs,lats




def plot_grid_image(grid,longs=None,lats=None,type_data='slp', _datetime=None, verbose=None, fileSaving=None,
                    vmin=None, vmax=None, title_add=None):
    '''
    plot the grid given, if there are the long/lats, the axis are set accordingly.
    '''
    grid = np.array(grid)
    if longs is not None: x_lims = [longs[0], longs[-1]]
    else:     x_lims = [0, len(grid[0])]
    if lats is not None:  y_lims = [lats[0], lats[-1]]
    else:     y_lims = [0, len(grid)]

    plt.figure()
    if not vmin: vmin=np.min(grid)
    if not vmax: vmax=np.max(grid)

    plt.imshow(grid, extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]],
               vmin=vmin, vmax=vmax,
               interpolation='nearest', origin='lower', cmap='seismic')  # cmap='hot',

    title = type_data
    if _datetime:  title=title+' '+str(_datetime)
    if title_add:  title=title+' '+str(title_add)
    plt.title(title)#, loc='left'
    cax = plt.axes([0.825, 0.1, 0.075, 0.80])
    plt.colorbar(cax=cax)
    if fileSaving:  plt.savefig(fileSaving)
    if verbose:     plt.show()


def get_windows_from_track_interim(track,instants,types, size_crop=10,
                               folder_data='./data/grid_1/sfc/sst_mont_pres_uvb_pv_crwc_sp/',
                                levtype='sfc' , folderLUT='./data/',
                                   levels=None, history=0):
    '''
    single track of a storm
    :return: grids corresponding to track
    '''
    units = 'hours since 1970-01-01 00:00:00 UTC'
    if history:
        for instant in instants:
            numdate=date2num(list_to_datetime(track.dates[instant]),units)
            new_datetime=num2date(numdate-6*history, units)
            track.dates[instant]=[new_datetime.year,new_datetime.month,new_datetime.day,new_datetime.hour]
    year=track.dates[0][0]
    month=track.dates[0][1]
    grids=[]
    rootgrp = open_netCDF_file(folder_data + "interim_daily_%04d%02d.nc" % (year, month))
    for instant  in instants:

        if track.dates[instant][1] != month:
            year = track.dates[instant][0]
            month = track.dates[instant][1]
            rootgrp = open_netCDF_file(folder_data + "interim_daily_%04d%02d.nc" % (year, month))

        center = [track.longitudes[instant], track.latitudes[instant]]
        a,b,c,netCDFnames=open_LUT_list_params(folderLUT+'list_params_nums_'+levtype+'.txt')
        types_netCDF=[netCDFnames[t] for t in types]
        grid, longs, lats=get_cropped_nefCDF_data_interim(rootgrp, params_netCDF=types_netCDF, _datetime=list_to_datetime(track.dates[instant]),
                                                  center=center, size=size_crop, params_shortnames=types, levels=levels)
        grids.append(grid)
    return grids


def get_distance_km(lon1, lat1, lon2, lat2):
    '''
    Using haversine formula (https://www.movable-type.co.uk/scripts/latlong.html)
    '''
    R=6371e3 # meters (earth's radius)
    phi_1=math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi=math.radians(lat2-lat1)
    delta_lambda=math.radians(lon2-lon1)
    a=np.power(math.sin(delta_phi/2),2) + math.cos(phi_1)*math.cos(phi_2)\
      * np.power(math.sin(delta_lambda/2),2)
    c= 2 * math.atan2(math.sqrt(a),math.sqrt(1-a))

    return R*c/1000.


def get_longlat_from_offsets(lon, lat, dkm_lon, dkm_lat):
    '''
    :param lon: initial longitude
    :param lat: inital latitude
    :param dn:     offsets in meters
    :param de:     offsets in meters
    :return: lon_final, lat_final
    '''
    # Earth’s radius, sphere
    R = 6378137
    # Coordinate offsets in radians
    dLat = dkm_lat / R
    dLon = dkm_lon / (R * math.cos(math.radians(lat)))

    # OffsetPosition, decimal degrees
    latO = lat + dLat * 180 / math.pi
    lonO = lon + dLon * 180 / math.pi

    return lonO, latO

##### ERA interim data ########
###############################

def load_ECMWF_server():
    server = ECMWFDataServer()
    return server


def interim_request(server, requestDates, requestParams, levtype, size_grid, targetfile, levelist=''):
    """
            An ERA interim request for analysis pressure level data.
        Change the keywords below to adapt it to your needs.
        (eg to add or to remove  levels, parameters, times etc)
        Request cost per day is 112 fields, 14.2326 Mbytes
    :param server: loaded server
    :param requestDates:
    :param requestParams:
    :param levtype: sfc, ml (model level), pl (pressure level), pt, pv
    :param size_grid:  if 1: 1 degree x 1 degree (0.75 is the smallest computed, but 0.25 is the smallest available)
    :param targetfile:
    :param levelist:
    :return:
    """
    if not levelist:
        levelist="0"

    server.retrieve({
        "class": "ei",
        "stream": "oper",
        "type": "an", # an= analysis, fc= forcast , 4v...
        "dataset": "interim", # interim= ERA-interim dataset
        "date": requestDates,
        "expver": "1",
        "levtype": levtype,
        "levelist": "/".join(map(str,levelist)),#"100/500/700/750/850/925/1000",
        "param": "/".join(requestParams),
        "target": targetfile,
        "time": "00/06/12/18",
        "format"    : "netcdf",
        "grid": str(size_grid)+"/"+str(size_grid)
    })


def retrieve_interim(yearStart=2000, yearEnd=2001, monthStart=1, monthEnd=12, list_params='',
                     levtype='sfc', size_grid=1,
                     targetfolder='./data/', levelist=''):
    '''
           A function to demonstrate how to iterate efficiently over several years and months etc
       for a particular interim_request.
    :param yearStart:
    :param yearEnd:
    :param monthStart:
    :param monthEnd:
    :param list_params: list of shortnames of the parameters wanted (get them from the LUT files
    :param levtype: sfc, ml (model level), pl (pressure level), pt, pv
    :param size_grid: if 1: 1 degree x 1 degree (0.75 is the smallest computed, but 0.25 is the smallest available)
    :param targetfolder: folder to store results and where are LUT files
    :param levelist: for pressure level or model level
    :return:
    '''
    curr_targetfolder=targetfolder+'grid_'+str(size_grid)+'/'+levtype+'/'+'_'.join(list_params)+'/'
    MMff.MakeDir(curr_targetfolder)
    server=load_ECMWF_server()

    requestParams_ids=get_params_ids_interim(list_params, levtype,
                           folderLUT=targetfolder)

    for year in list(range(yearStart, yearEnd + 1)):
        for month in list(range(monthStart, monthEnd + 1)):
            startDate = '%04d%02d%02d' % (year, month, 1)
            numberOfDays = calendar.monthrange(year, month)[1]
            lastDate = '%04d%02d%02d' % (year, month, numberOfDays)
            targetfile = curr_targetfolder+"interim_daily_%04d%02d.nc" % (year, month)
            requestDates = (startDate + "/TO/" + lastDate)
            interim_request(server, requestDates, requestParams_ids, levtype, size_grid, targetfile, levelist)



def open_LUT_list_params(LUTfilename):
    '''
    the Look up table files are here to help mapping the parameter names to its number used in the netCDF files.
    :param LUTfilename: total path of the .txt (ex: path_to/list_params_nums_ml.txt)
    :return: 3 lists: shortnames, corresponding ids, total names of the parameters
    '''
    shortnames=[]; ids={}; names={}; names_netCDF={}
    with open(LUTfilename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            shortnames.append(row['shortname'][1:])
            ids[row['shortname'][1:]]=row['id']
            names[row['shortname'][1:]]=row['name'][1:]
            names_netCDF[row['shortname'][1:]]=row['CDFname'][1:]

    return shortnames, ids, names, names_netCDF


def open_level_list(levtype='ml', folderLUT='./data/'):
    filename=folderLUT+'list_levels_'+levtype+'.txt'
    with open(filename, newline='') as file:
        level_list=file.read().splitlines()
    return level_list


def get_params_ids_interim(list_params,levtype,
                           folderLUT='./data/'):
    '''
    Get the numbers of the parameters wanted to collect in the ERA interim database, using the LUT files.
    :param list_params:
    :param levtype:
    :param folderpath:
    :return:
    '''
    LUTfilename=folderLUT+'list_params_nums_'+str(levtype)+'.txt'
    shortnames, ids, names, namesCDF = open_LUT_list_params(LUTfilename)
    params_ids=[]
    for param in list_params:
        params_ids.append(ids[param])

    return params_ids

def get_center_value_Xdata(X):
    '''
    keep only the central value of the images,
    :param X: array size nb_samples x nb_params x size_crop x size_crop (size_crop is longitude/latitude)
    :return: X_center, array size nb_samples x nb_params
    '''
    size_crop=len(X[0][0])
    ncenter=int((size_crop-1)/2)
    print('center:'+str(ncenter), flush=True)
    Xcenters=np.zeros([len(X),len(X[0])])
    for i,x in enumerate(X):
        Xcenters[i]=[xparam[ncenter][ncenter] for xparam in x]

    return Xcenters

def get_mean_value_Xdata(X):
    '''
    keep only the central value of the images,
    :param X: array size nb_samples x nb_params x size_crop x size_crop (size_crop is longitude/latitude)
    :return: X_center, array size nb_samples x nb_params
    '''
    Xmeans=np.zeros([len(X),len(X[0])])
    for i,x in enumerate(X):
        Xmeans[i]=[np.mean(xparam) for xparam in x]

    return Xmeans

def get_deriv_values_Xdata(X,dist_diff=2, flag_center=False):
    '''
    get the 4 derivatives of the parameters in +-longitude, +-latitude wrt the center.
    :param X: array size nb_samples x nb_params x size_crop x size_crop (size_crop is longitude/latitude)
            flag_center: if True, adds the center value at the end.
    :return: Xderiv, array size nb_samples x nb_params x 4
    '''
    size_crop=len(X[0][0])
    ncenter=int((size_crop-1)/2)
    if dist_diff>ncenter:
        print('Warning! distance to center for the derivative is too large. setting it to '+str(ncenter), flush=True)
        dist_diff=ncenter
    if flag_center:
        Xderiv = np.zeros([len(X), len(X[0]), 5])
    else: Xderiv = np.zeros([len(X), len(X[0]), 4])
    for i,x in enumerate(X):
        for p,xparam in enumerate(x):
            Xderiv[i][p][0] = xparam[ncenter][ncenter] - xparam[int(ncenter+dist_diff)][ncenter]
            Xderiv[i][p][1] = xparam[ncenter][ncenter] - xparam[int(ncenter - dist_diff)][ncenter]
            Xderiv[i][p][2] = xparam[ncenter][ncenter] - xparam[ncenter][int(ncenter + dist_diff)]
            Xderiv[i][p][3] = xparam[ncenter][ncenter] - xparam[ncenter][int(ncenter - dist_diff)]
            if flag_center:
                Xderiv[i][p][4] = xparam[ncenter][ncenter]

    return Xderiv


def crop_grids(X,crop_final=11):
    size_crop=len(X[0][0])
    if crop_final>size_crop:
        print('desired crop is larger than initial.')
        return X
    elif crop_final==size_crop:
        return X
    ncenter=int((size_crop-1)/2)
    nmin=int(ncenter-(crop_final-1)/2)
    print(nmin)
    nmax=int(ncenter+(crop_final-1)/2+1)
    print(nmax)
    Xsmall=np.zeros([len(X),len(X[0]),crop_final,crop_final])
    for i,x in enumerate(X):
        Xsmall[i]=[np.array(xparam)[nmin:nmax,nmin:nmax] for xparam in x]
    return Xsmall
