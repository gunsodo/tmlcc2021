import pandas as pd
import numpy as np
import math

def clean(filename='train.csv', method=['rm', 'mean', 'median'], remove_MOFname=False, remove_func_group_NAN=True, save=True):
    '''
    filename: csv file
    method: rm for remove, mean, and median
    remove_MOFname: True to remove the first column of the CSV file
    save: True to save file
    '''
    
    df_ori = pd.read_csv(filename)
    drop_subset = []
    
    if remove_MOFname:
        del df_ori['MOFname']
        
    if remove_func_group_NAN:
        drop_subset.append('functional_groups')
    
    if 'rm' in method:
        # remove corrupted data
        df = df_ori.copy()
        df.replace(0, np.nan, inplace=True)
        df.replace(-1, np.nan, inplace=True)
        df.replace(math.inf, np.nan, inplace=True)
        savename = filename[:-4] + '_rm'
        if remove_func_group_NAN:
            df.dropna(axis=0, how='any', inplace=True)
        else:
            df.dropna(axis=0, how='any', subset=[i for i in df.columns if i != 'functional_groups'], inplace=True)
            savename += '_keepfg'
        if save:
            df.to_csv(savename + '.csv', index=False)
            print(savename + '.csv is saved')
    
    if 'mean' in method:
        # use mean for corrupted data
        df = df_ori.copy()
        df.dropna(subset=drop_subset, inplace=True)
        df.replace(0, np.nan, inplace=True)
        df.replace(-1, np.nan, inplace=True)
        df.replace(math.inf, np.nan, inplace=True)
        mean = df.mean(axis=0)
        df.fillna(mean, inplace=True)
        savename = filename[:-4] + '_mean'
        if not remove_func_group_NAN:
            savename += '_keepfg'
        if save:
            df.to_csv(savename + '.csv', index=False)
            print(savename + '.csv is saved')
        
    if 'median' in method:
        # use median for corrupted data
        df = df_ori.copy()
        df.dropna(subset=drop_subset, inplace=True)
        df.replace(0, np.nan, inplace=True)
        df.replace(-1, np.nan, inplace=True)
        df.replace(math.inf, np.nan, inplace=True)
        median = df.median(axis=0)
        df.fillna(median, inplace=True)
        savename = filename[:-4] + '_median'
        if not remove_func_group_NAN:
            savename += '_keepfg'
        if save:
            df.to_csv(savename + '.csv', index=False)
            print(savename + '.csv is saved')
    
    if 'nan' in method:
        # just replace corrupted data with nan
        # remove func_group will not affect this function
        df = df_ori.copy()
        df.replace(0, np.nan, inplace=True)
        df.replace(-1, np.nan, inplace=True)
        df.replace(math.inf, np.nan, inplace=True)
        savename = filename[:-4] + '_nan'
        if save:
            df.to_csv(savename + '.csv', index=False)
            print(savename + '.csv is saved')

    # TODO add more method
