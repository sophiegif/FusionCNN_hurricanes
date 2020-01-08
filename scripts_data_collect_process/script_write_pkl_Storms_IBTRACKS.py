# Copyright (c) 2020 Sophie Giffard-Roisin <sophie.giffard@univ-grenoble-alpes.fr>
# SPDX-License-Identifier: GPL-3.0

import DataProcessing.ModuleStormReader as Msr

# get this file or a similar one from https://www.ncdc.noaa.gov/ibtracs/
namefile = './data/Allstorms.ibtracs_wmo.v03r10.nc'

date_init = 1979
namefilesaving = './data' + 'tracks_'+str(date_init)+'_after_full_data.pkl'
flag_full_data = True

Msr.save_all_storm_tracks_IBTRACKS(namefile, namefilesaving,
                                   date_init=date_init, flag_full_data=flag_full_data)
