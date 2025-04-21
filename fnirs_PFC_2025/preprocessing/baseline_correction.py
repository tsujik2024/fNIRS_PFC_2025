"""
Implements baseline subtraction for fNIRS data, using either:
- A custom baseline DataFrame (user-provided), or
- The quiet stance period between events 'S1' and 'S2' in the data.

Columns like 'Sample number', 'Event', and 'Time (s)' are ignored.
"""

import pandas as pd

def baseline_subtraction(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    baseline_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Applies baseline subtraction to the given DataFrame of fNIRS signals.

    If a `baseline_df` is provided, the baseline mean is computed from that DataFrame
    (for each channel) and subtracted from `df`. Otherwise, the baseline is computed
    from the time interval between events 'S1' and 'S2' in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fNIRS data (columns for channels), plus any metadata
        columns like 'Sample number', 'Event', 'Time (s)' that should be ignored.
    events_df : pd.DataFrame
        DataFrame specifying at least three events (S1, S2, S3). If no `baseline_df`
        is provided, this script looks for 'S1' and 'S2' to define the baseline period.
        Must have columns:
          - 'Sample number'
          - 'Event'
    baseline_df : pd.DataFrame, optional
        If provided, each channel's baseline mean is computed from this DataFrame
        instead of from S1->S2 in `df`. Must have the same column names as `df`.
        Default is None.

    Returns
    -------
    corrected_df : pd.DataFrame
        A new DataFrame with the baseline-subtracted signals.

    Raises
    ------
    ValueError
        If the required events ('S1', 'S2') are not found,
        or if their sample indices are out of range,
        or if `events_df` does not have exactly 3 events and no custom baseline is provided.
    """
    corrected_df = df.copy()

    # Identify which columns are channels vs. metadata
    ignore_cols = ['Sample number', 'Event', 'Time (s)']
    data_cols = [col for col in corrected_df.columns if col not in ignore_cols]

    if baseline_df is not None:
        # ------------------------------
        # Use the provided baseline_df
        # ------------------------------
        for ch in data_cols:
            baseline_mean = baseline_df[ch].mean()
            corrected_df[ch] = corrected_df[ch] - baseline_mean

    else:
        # ---------------------------------------------------
        # Compute baseline from quiet stance (S1 -> S2) in df
        # ---------------------------------------------------
        if len(events_df) != 3:
            raise ValueError(
                f"The number of events in events_df is {len(events_df)}, expected 3. "
                "When no custom baseline_df is provided, we need exactly three events, "
                "with S1 and S2 defining the baseline window."
            )

        # Extract sample numbers for S1 and S2
        if 'S1' not in events_df['Event'].values or 'S2' not in events_df['Event'].values:
            raise ValueError("Events 'S1' and/or 'S2' are missing from events_df.")

        s1_sample = events_df.loc[events_df['Event'] == 'S1', 'Sample number'].values[0]
        s2_sample = events_df.loc[events_df['Event'] == 'S2', 'Sample number'].values[0]

        start = int(s1_sample)
        end = int(s2_sample)

        # Check if 'start' and 'end' are within bounds
        if not (0 <= start < len(corrected_df)) or not (0 < end <= len(corrected_df)):
            raise ValueError(
                f"Event indices out of bounds: start={start}, end={end}, "
                f"data length={len(corrected_df)}"
            )
        if start >= end:
            raise ValueError(
                f"The baseline interval is invalid: S1={start} >= S2={end}."
            )

        # Subtract the mean during the baseline period
        for ch in data_cols:
            baseline_segment = corrected_df.loc[start:end, ch]
            baseline_mean = baseline_segment.mean()
            corrected_df[ch] = corrected_df[ch] - baseline_mean

    return corrected_df
