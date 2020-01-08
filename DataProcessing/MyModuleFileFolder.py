# Copyright (c) 2020 Sophie Giffard-Roisin <sophie.giffard@univ-grenoble-alpes.fr>
# SPDX-License-Identifier: GPL-3.0

import os

def MakeDir(folderSaving):
    try:
        os.makedirs(folderSaving)
    except OSError:
        if not os.path.isdir(folderSaving):
            raise

def modify_folder_flag(foldername,flagname,flagvalue):
    if flagvalue:
        foldername=foldername[:-1]+flagname[4:]+'/'
    return foldername
