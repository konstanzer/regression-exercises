import pandas as pd
import numpy as np
from scipy import stats


def wrangle_zillow(df):
    
    df['yearbuilt'].fillna(np.nanmedian(zillow.yearbuilt), inplace=True)
    df['taxamount'].fillna(np.nanmedian(zillow.taxamount), inplace=True)
    df['calculatedfinishedsquarefeet'].fillna(np.nanmedian(zillow.calculatedfinishedsquarefeet), inplace=True)
    
    df['bedroomcnt'].replace(0, np.median(zillow.bedroomcnt), inplace=True)
    df['bathroomcnt'].replace(0, np.median(zillow.bathroomcnt), inplace=True)

    # drop row if at least one column has a z-score over 3
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    
    df = df.astype('int')
    
    returnn df