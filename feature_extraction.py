import os
import numpy as np
import pandas as pd


# Helper functions.
def zero_crossing(x):
    """
    Count the number of times the signal value changed signs.
    """
    return sum((x.iloc[:-1] * x.shift(-1).iloc[:-1]) < 0)


def percentile(p):
    """
    Helper function to compute percentile p.
    """
    def percentile_(x):
        return np.percentile(x, p)
    percentile_.__name__ = 'percentile_%s' % p
    return percentile_


def add_mean_last3(w1_df, w10_df):
    """
    Compute mean of last 3 seconds from each 10-second window and join back
    to w10 dataframe.
    """
    new = w1_df.groupby(['pid', 'window10']).tail(3).groupby(['pid', 'window10']).mean().reset_index()\
        .drop(['window1'], axis=1)
    new.columns = ['_'.join([col, 'last3']) for col in new.columns.values]
    new = new.rename(columns={'pid_last3': 'pid',
                              'window10_last3': 'window10'})
    return pd.merge(w10_df, new, how='left', on=['pid', 'window10'])


def add_mean_first3(w1_df, w10_df):
    """
    Compute mean of first 3 seconds from each 10-second window 
    and join back to w10 dataframe.
    """
    new = w1_df.groupby(['pid', 'window10']).head(3).groupby(['pid', 'window10']).mean().reset_index()\
        .drop(['window1'], axis=1)
    new.columns = ['_'.join([col, 'first3']) for col in new.columns.values]
    new = new.rename(columns={'pid_first3': 'pid',
                              'window10_first3': 'window10'})
    return pd.merge(w10_df, new, how='left', on=['pid', 'window10'])


# Windowing functions.
def pivot_window_10s_from_ms(df):
    """
    Given millisecond-level data, compute 'mean', median', 'min', 'max', 'std', 
    percentiles, and zero-crossing per 10-second window.
    Pivot into a single row (uniquely identified by window10-pid).
    """
    df['window10'] = np.floor(df['time'] / 10000).astype(int)
    df = df.groupby(['pid', 'window10'])[['x', 'y', 'z']]\
        .agg(['mean', 'median', 'min', 'max', 'std',
              percentile(5), percentile(25), percentile(75), percentile(95), zero_crossing])
    df.columns = ['_'.join([str(c) for c in col]).strip()
                  for col in df.columns.values]
    df = df.reset_index()
    return df.reset_index()


def pivot_window_1s(df):
    """
    Compute 'mean', median', 'min', 'max', 'std' per 1-second window per pid 
    and pivot into a single row (uniquely identified by window1-pid).
    Input df columns: ['x', 'y', 'z']
    Output df columns: ['x_median', 'x_min',...'z_median']
    """
    df['window1'] = np.floor(df['time'] / 1000).astype(int)
    df = df.groupby(['pid', 'window1'])[['x', 'y', 'z']]\
        .agg(['mean', 'median', 'min', 'max', 'std'])
    df.columns = ['_'.join([str(c) for c in col]).strip()
                  for col in df.columns.values]
    return df.reset_index()


def pivot_window_10s_from_1s(df):
    """
    Calls pivot_window_1s to compute 1-second window metrics.
    Compute 'mean', median', 'min', 'max', 'std', 'first3_mean', 'last3_mean',
    of computed 1-second window metrics per 10-second window per pid 
    and pivot into a single row (uniquely identified by window10-pid).
    Input df columns: ['x_median', 'y_median','z_median',...]
    Output df columns: ['x_median_mean', 'y_median_mean', 'z_median_mean', 'x_median_median',...]
    """
    w1 = pivot_window_1s(df)
    w1['window10'] = np.floor(w1['window1'] / 10).astype(int)
    two_tier_df = w1.groupby(['pid', 'window10'])[w1.drop(['pid', 'window1', 'window10'], axis=1).columns]\
        .agg(['mean', 'median', 'min', 'max', 'std'])
    two_tier_df.columns = ['_'.join([str(c) for c in col]).strip()
                           for col in two_tier_df.columns.values]
    two_tier_df = two_tier_df.reset_index()
    # Compute mean of first and last 3 seconds within the 10-second window.
    two_tier_df = add_mean_last3(w1, two_tier_df)
    two_tier_df = add_mean_first3(w1, two_tier_df)
    # Impute nan standard deviation (when window10 is a single row) as 0.
    two_tier_df = two_tier_df.fillna(0)
    return two_tier_df


def two_tier_windowing(df):
    """
    Run single and two-tier windowing functions 
    and merge generated features together.
    Returns a dataframe with all features.
    """
    single_tier = pivot_window_10s_from_ms(df)
    two_tier = pivot_window_10s_from_1s(df)
    return pd.merge(single_tier, two_tier, how='left', on=['pid', 'window10'])


def run_feature_engineering(acc_path):
    """
    Load each preprocessed accelerometer file and
    create all features using two-tiered windowing.
    
    Returns a concatenated dataframe with
    accelerometer data for all participants.
    """
    dfs = []
    directory = os.fsencode(acc_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename != '.DS_Store':
            print(filename)
            df = pd.read_pickle(acc_path + filename)
            df = two_tier_windowing(df)
            dfs.append(df)
    return pd.concat(dfs).reset_index().drop(columns=['level_0', 'index'], axis=1)


# Joining target to features.
def reconcile_acc_tac(acc, tac):
    """
    Merge target "intoxicated" variable onto windowed accelerometer df by taking the most 
    recent target value where tac timestamp (10s window) <= acc timestamp (10s window)
    for a given pid.
    """
    # Create window10 timestamp on tac df.
    tac['window10'] = np.floor(tac['timestamp'] / 10).astype(np.int64)
    # Sort both df by window10.
    acc = acc.sort_values(['window10'], ascending=True)
    tac = tac.sort_values(['window10'], ascending=True)
    # Merge the last row in tac whose tac timestamp <= to the acc timestamp.
    return pd.merge_asof(acc, tac, on='window10', by='pid').reset_index(drop=True)


## Code from https://github.com/lisachua/detect_heavy_drinking
