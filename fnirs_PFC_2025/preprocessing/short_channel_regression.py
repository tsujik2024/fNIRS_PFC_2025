"""
Implements short channel correction (short channel regression) to remove
superficial components from long-channel fNIRS signals.

References:
    - Gagnon et al., 2014
    - Brigadoi et al., 2014
"""

import pandas as pd
import numpy as np

def scr_regression(long_data: pd.DataFrame, short_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply short channel correction to remove the superficial (skin blood flow) component
    from long-channel fNIRS measurements.

    A simple linear regression is performed between the mean of short channels (predictor)
    and each long channel (outcome). The resulting fit is then subtracted from the long channel,
    effectively removing the component common to short channels.

    Parameters
    ----------
    long_data : pd.DataFrame
        DataFrame containing fNIRS data (columns) for the long channels.
        Each column is typically something like "CH1 HbO", "CH1 HbR", etc.
        Rows represent timepoints/samples.
    short_data : pd.DataFrame
        DataFrame containing fNIRS data (columns) for the short reference channels,
        measured at superficial depths. Each column is typically "CHx HbO"/"CHx HbR"
        for short-separation channels.

    Returns
    -------
    long_data_corrected : pd.DataFrame
        DataFrame with the same shape and columns as `long_data`,
        but after subtracting the short-channel component.
    """
    # Copy to avoid mutating the original data
    long_data_corrected = long_data.copy()

    # Take the mean across short channels at each timepoint
    # resulting in a 1D array of length n_samples
    short_mean = short_data.mean(axis=1)

    # Linear regression for each long channel:
    #   Y_long = Y_long - beta * short_mean
    # where beta = (X^T Y) / (X^T X), X = short_mean, Y = each column in long_data.
    for col in long_data_corrected.columns:
        Y = long_data[col].values
        X = short_mean.values

        # Compute beta via dot product
        denom = np.dot(X, X)
        if denom == 0:
            # If short_mean is all zeros (or extremely close), skip correction
            # to avoid division by zero
            beta = 0.0
        else:
            beta = np.dot(X, Y) / denom

        # Subtract the superficial component
        long_data_corrected[col] = Y - beta * X

    return long_data_corrected
