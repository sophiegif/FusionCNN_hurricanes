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