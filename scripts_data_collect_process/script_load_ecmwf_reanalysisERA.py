# Copyright (c) 2020 Sophie Giffard-Roisin <sophie.giffard@univ-grenoble-alpes.fr>
# SPDX-License-Identifier: GPL-3.0

import DataProcessing.ModuleReanalysisData as Mre

yearStart = 2009
yearEnd = 2010
monthStart = 1
monthEnd = 12

folder_saving = './data/'
size_grid = 1
levtype = 'pl'

possible_levels = Mre.open_level_list('pl', folderLUT=folder_saving)

levelist = [lev for lev in possible_levels if int(lev) > 90] # no level smaller than 100 hPa
levelist = levelist[1::2] # take only half of the levels (random choice)

LUTfilename = folder_saving + 'list_params_nums_'+levtype+'.txt'
shortnames, ids, names, namesCDF = Mre.open_LUT_list_params(LUTfilename)

#list_params=['sst', 'mont', 'pres', 'uvb', 'pv', 'crwc', 'sp']
list_params = shortnames

# check if params in list params:
if not set(list_params).issubset(shortnames):
    print('Warning! list_params is not a subset of possible parameters!')

#check if levelist in possible levels:
if levelist:
    possible_levels = Mre.open_level_list('pl', folderLUT=folder_saving)
    if not set(levelist).issubset(possible_levels):
        print('Warning! levelist is not a subset of possible levels!')

Mre.retrieve_interim(yearStart=yearStart, yearEnd=yearEnd, monthStart=monthStart,
                     monthEnd=monthEnd, list_params=list_params,
                     levtype=levtype, size_grid=size_grid,
                     targetfolder=folder_saving, levelist=levelist)
