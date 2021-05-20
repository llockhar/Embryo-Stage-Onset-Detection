import numpy as np
import random

def split_folders(folders, fold):
    if fold == 0:
        random.Random(433).shuffle(folders)
        folders_train = folders[:round(0.70*len(folders))]  
        folders_val = folders[round(0.70*len(folders)):round(0.85*len(folders))]  
        folders_test = folders[round(0.85*len(folders)):]   
    else:
        folders_train, folders_val, folders_test = roll_data(folders)
    
    return folders_train, folders_val, folders_test

def roll_data(folders):
    folders = np.asarray(folders)
    rolled_folders = np.roll(folders, round(len(folders)/5), axis=0)
    rolled_folders = list(rolled_folders)
    
    folders_train = rolled_folders[:round(0.70*len(folders))]
    folders_val = rolled_folders[round(0.70*len(folders)):round(0.85*len(folders))]
    folders_test = rolled_folders[round(0.85*len(folders)):]

    return folders_train, folders_val, folders_test


