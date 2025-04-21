import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_channels_separately(data, fs, title="Channel Signals",
                             subject=None, condition=None, y_lim=None):
    """
    Plot each channel with consistent Y-axis scaling across subjects

    Parameters:
    ----------
    data : DataFrame
        Data to plot
    fs : float
        Sampling frequency
    title : str
        Plot title
    subject : str, optional
        Subject identifier
    condition : str, optional
        Condition label
    y_lim : tuple, optional
        Y-axis limits as (min, max). If None, will use the auto-scaled limits.
    """
    if hasattr(data, "columns"):
        channels = {}
        for col in data.columns:
            parts = col.split()
            if len(parts) < 2:
                continue
            ch_id = parts[0]
            signal_type = parts[1]
            if ch_id not in channels:
                channels[ch_id] = {}
            channels[ch_id][signal_type] = data[col]
        channel_ids = sorted(channels.keys())
        time = np.arange(len(data)) / fs
        fig, axes = plt.subplots(nrows=len(channel_ids), ncols=1,
                                 figsize=(10, 3 * len(channel_ids)), sharex=True)
        if len(channel_ids) == 1:
            axes = [axes]
        title_parts = [title]
        if subject: title_parts.append(f"Subject: {subject}")
        if condition: title_parts.append(f"({condition})")
        fig.suptitle("\n".join(title_parts))

        # If y_lim is None, calculate global min and max across all channels
        if y_lim is None:
            all_values = []
            for ch in channel_ids:
                for key in channels[ch]:
                    if key in ["O2Hb", "HbO", "HHb", "HbR"]:
                        all_values.extend(channels[ch][key].values)
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                # Add a small buffer (5% of range)
                buffer = 0.05 * (max_val - min_val)
                y_lim = (min_val - buffer, max_val + buffer)

        for i, ch in enumerate(channel_ids):
            ax = axes[i]

            # Plot HbO
            for o2_key in ["O2Hb", "HbO"]:
                if o2_key in channels[ch]:
                    ax.plot(time, channels[ch][o2_key], 'r-', label=f'{ch} {o2_key}')
                    break

            # Plot HbR
            for hb_key in ["HHb", "HbR"]:
                if hb_key in channels[ch]:
                    ax.plot(time, channels[ch][hb_key], 'b-', label=f'{ch} {hb_key}')
                    break

            # Set consistent y-axis limits if provided
            if y_lim is not None:
                ax.set_ylim(y_lim)

            ax.set_ylabel("Δ Concentration (mmol/L)")
            ax.legend(loc='upper right')
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        return fig, axes, y_lim  # Return y_lim so it can be reused


def plot_overall_signals(data, fs, title="Overall Signals",
                         subject=None, condition=None, y_lim=None):
    """
    Plot overall signals with consistent Y-axis scaling

    Parameters:
    ----------
    data : DataFrame
        Data to plot
    fs : float
        Sampling frequency
    title : str
        Plot title
    subject : str, optional
        Subject identifier
    condition : str, optional
        Condition label
    y_lim : tuple, optional
        Y-axis limits as (min, max). If None, will use the auto-scaled limits.
    """
    fig = plt.figure(figsize=(12, 5))

    # Build title
    title_parts = [title]
    if subject: title_parts.append(f"Subject: {subject}")
    if condition: title_parts.append(f"({condition})")
    plt.title("\n".join(title_parts))

    # If y_lim is None, calculate global min and max
    if y_lim is None:
        all_values = []
        for o2_col in ["Mean HbO", "grand oxy"]:
            if o2_col in data.columns:
                all_values.extend(data[o2_col].values)

        for hb_col in ["Mean HHb", "grand deoxy"]:
            if hb_col in data.columns:
                all_values.extend(data[hb_col].values)

        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            # Add a small buffer (5% of range)
            buffer = 0.05 * (max_val - min_val)
            y_lim = (min_val - buffer, max_val + buffer)

    # Plot HbO
    for o2_col in ["Mean HbO", "grand oxy"]:
        if o2_col in data.columns:
            time = data.get("Time (s)", np.arange(len(data)) / fs)
            plt.plot(time, data[o2_col], 'r-', label='HbO')
            break

    # Plot HbR
    for hb_col in ["Mean HHb", "grand deoxy"]:
        if hb_col in data.columns:
            time = data.get("Time (s)", np.arange(len(data)) / fs)
            plt.plot(time, data[hb_col], 'b-', label='HHb')
            break

    # Set consistent y-axis limits if provided
    if y_lim is not None:
        plt.ylim(y_lim)

    plt.xlabel("Time (s)")
    plt.ylabel("Δ Concentration (mmol/L)")
    plt.legend()
    plt.tight_layout()

    return fig, y_lim  # Return y_lim so it can be reused


def calculate_global_ylim(data_list, include_cols=None):
    """
    Calculate global y-axis limits across multiple datasets

    Parameters:
    ----------
    data_list : list of DataFrames
        List of dataframes to analyze
    include_cols : list of str, optional
        List of column substring patterns to include. If None, will use default patterns.

    Returns:
    -------
    tuple
        (min_value, max_value) for y-axis limits
    """
    if include_cols is None:
        include_cols = ["O2Hb", "HbO", "HHb", "HbR", "Mean HbO", "Mean HHb", "grand oxy", "grand deoxy"]

    all_values = []

    for data in data_list:
        if not hasattr(data, "columns"):
            continue

        for col in data.columns:
            if any(pattern in col for pattern in include_cols):
                all_values.extend(data[col].dropna().values)

    if not all_values:
        return None

    min_val = min(all_values)
    max_val = max(all_values)

    # Add a small buffer (5% of range)
    buffer = 0.05 * (max_val - min_val)

    return (min_val - buffer, max_val + buffer)