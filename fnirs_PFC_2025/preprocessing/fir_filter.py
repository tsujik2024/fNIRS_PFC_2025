import pandas as pd
import numpy as np
from scipy.signal import firwin, filtfilt

def fir_filter(df: pd.DataFrame, order: int, Wn: list, fs: int):
    filtered_df = df.copy()
    data_columns = [col for col in df.columns if col not in ['Sample number', 'Event']]
    for ch in data_columns:
        ch_asarray = np.array(df[ch], dtype='float64')
        b = firwin(order + 1, Wn, pass_zero=False, fs=fs)
        ch_filtered = filtfilt(b, [1.0], ch_asarray)
        filtered_df[ch] = ch_filtered
    return filtered_df
