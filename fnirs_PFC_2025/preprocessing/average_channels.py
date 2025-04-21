"""
averages multiple fNIRS channels into hemisphere-level signals:
- Left hemisphere channels (e.g., CH4, CH5, CH6)
- Right hemisphere channels (e.g., CH1, CH2, CH3)

Creates columns for O2Hb (oxy) and HHb (deoxy) on each hemisphere,
plus "grand" averages across all channels.

Parameters:
    channels_to_exclude: optional list of channel numbers to exclude.
"""

import pandas as pd
import numpy as np

def average_channels(df: pd.DataFrame, channels_to_exclude=None) -> pd.DataFrame:
    """
    Averages multiple fNIRS channels into hemisphere-level and grand-mean signals.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing columns like:
          - "CH{ch} HbO" and "CH{ch} HbR" for channel ch
          - 'Sample number'
          - 'Event'
        The script will create new columns for left/right hemisphere oxy/deoxy
        averages, as well as grand oxy/deoxy across left+right hemispheres.
    channels_to_exclude : list of int, optional
        A list of channel indices (e.g., [1,2]) to be excluded from the averaging.

    Returns
    -------
    ret_df : pd.DataFrame
        A DataFrame with columns:
          - 'Sample number'
          - 'left oxy', 'left deoxy'
          - 'right oxy', 'right deoxy'
          - 'grand oxy', 'grand deoxy'
          - 'Event'
        The row indices match those of the input df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Must provide a DataFrame, not {type(df)}.")

    channels_to_exclude = channels_to_exclude or []
    df_copy = df.copy()

    # Define hemisphere channel numbers (1-based indexing)
    left_channels = [4, 5, 6]
    right_channels = [1, 2, 3]

    # Remove any channels we want to exclude
    left_channels = [ch for ch in left_channels if ch not in channels_to_exclude]
    right_channels = [ch for ch in right_channels if ch not in channels_to_exclude]

    # Identify columns for each hemisphere
    left_hbo_cols = [f'CH{ch} HbO' for ch in left_channels if f'CH{ch} HbO' in df_copy.columns]
    left_hbr_cols = [f'CH{ch} HbR' for ch in left_channels if f'CH{ch} HbR' in df_copy.columns]
    right_hbo_cols = [f'CH{ch} HbO' for ch in right_channels if f'CH{ch} HbO' in df_copy.columns]
    right_hbr_cols = [f'CH{ch} HbR' for ch in right_channels if f'CH{ch} HbR' in df_copy.columns]

    # Compute means across columns for each hemisphere
    left_oxy = df_copy[left_hbo_cols].mean(axis=1) if left_hbo_cols else np.nan
    left_deoxy = df_copy[left_hbr_cols].mean(axis=1) if left_hbr_cols else np.nan
    right_oxy = df_copy[right_hbo_cols].mean(axis=1) if right_hbo_cols else np.nan
    right_deoxy = df_copy[right_hbr_cols].mean(axis=1) if right_hbr_cols else np.nan
    grand_oxy = df_copy[left_hbo_cols + right_hbo_cols].mean(axis=1) if (left_hbo_cols or right_hbo_cols) else np.nan
    grand_deoxy = df_copy[left_hbr_cols + right_hbr_cols].mean(axis=1) if (left_hbr_cols or right_hbr_cols) else np.nan

    # Build the return DataFrame
    ret_cols = {}

    # Optional columns
    if 'Sample number' in df_copy.columns:
        ret_cols['Sample number'] = df_copy['Sample number']
    if 'Event' in df_copy.columns:
        ret_cols['Event'] = df_copy['Event']

    # Averages
    ret_cols['left oxy'] = left_oxy
    ret_cols['left deoxy'] = left_deoxy
    ret_cols['right oxy'] = right_oxy
    ret_cols['right deoxy'] = right_deoxy
    ret_cols['grand oxy'] = grand_oxy
    ret_cols['grand deoxy'] = grand_deoxy

    ret_df = pd.DataFrame(ret_cols, index=df_copy.index)

    return ret_df
