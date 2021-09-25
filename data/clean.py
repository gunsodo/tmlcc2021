import pandas as pd
import numpy as np
import math

def clean(filename='train.csv', method=['rm', 'mean', 'median'], remove_MOFname=False, save=True):
    '''
    filename: csv file
    method: rm for remove, mean, and median
    remove_MOFname: True to remove the first column of the CSV file
    save: True to save file
    '''
    
    df = pd.read_csv(filename)
    
    if remove_MOFname:
        del df['MOFname']
        
    if 'rm' in method:
        # remove corrupted data
        df.replace(0, np.nan, inplace=True)
        df.replace(-1, np.nan, inplace=True)
        df.replace(math.inf, np.nan, inplace=True)
        df.dropna(axis=0, how='any', inplace=True)
        print(filename[:-4] + '_rm.csv')
        if save:
            df.to_csv(filename[:-4] + '_rm.csv')
    
    if 'mean' in method:
        # use mean for corrupted data
        df.dropna(subset=['functional_groups', 'topology'], inplace=True)
        df.replace(0, np.nan, inplace=True)
        df.replace(-1, np.nan, inplace=True)
        df.replace(math.inf, np.nan, inplace=True)
        mean = df.mean(axis=0)
        df.fillna(mean, inplace=True)
        print(df)
        if save:
            df.to_csv(filename[:-4] + '_mean.csv')
        
    if 'median' in method:
        # use median for corrupted data
        df = pd.read_csv('train.csv')
        df.dropna(subset=['functional_groups', 'topology'], inplace=True)
        df.replace(0, np.nan, inplace=True)
        df.replace(-1, np.nan, inplace=True)
        df.replace(math.inf, np.nan, inplace=True)
        median = df.median(axis=0)
        df.fillna(median, inplace=True)
        print(df)
        if save:
            df.to_csv(filename[:-4] + '_median.csv')
            
    # TODO add more method