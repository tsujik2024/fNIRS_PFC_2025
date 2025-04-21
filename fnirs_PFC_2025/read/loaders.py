import pandas as pd
import numpy as np
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust logging level as needed


def read_txt_file(file_path: str) -> dict:
    """
    Parse a .txt export of fNIRS data generated in Oxysoft, with additional checks
    and logging statements to handle unexpected file structure.

    :param file_path: path to raw data file
    :return: dictionary of metadata and raw fnirs data in the format:
        {
            'metadata': { ... },
            'data': pd.DataFrame  # columns are renamed to CH0 HbO, CH0 HbR, CH1 HbO, CH1 HbR, etc.
        }
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.read().split('\n')
    except Exception as e:
        logger.error(f"Failed to open/read file '{file_path}'. Error: {e}")
        raise

    if not lines or len(lines) == 0:
        msg = f"File '{file_path}' is empty or could not be read properly."
        logger.error(msg)
        raise IOError(msg)

    # Split each line into columns
    rows = [row.split('\t') for row in lines]

    # Extract metadata and data from rows
    metadata = _read_metadata(rows, file_path)
    df = _read_data(rows, file_path)

    # Debug: log what columns we have before renaming
    logger.debug(f"Columns before renaming in '{file_path}': {df.columns.tolist()}")

    # --- Reassign column names into paired channels ---
    df = _reassign_channels(df, file_path)

    return {'metadata': metadata, 'data': df}


def _read_metadata(rows: list, file_path: str) -> dict:
    """
    Internal helper function to parse the header lines and return metadata.
    Expects to find info in the first ~7 lines of the .txt file.
    """
    rows_copy = rows.copy()
    metadata = {}

    lines_to_check = min(7, len(rows_copy))
    found_oxysoft_export_line = False

    for i in range(lines_to_check):
        row = rows_copy[i]
        if not row:
            logger.warning(
                f"Line {i} in the file header is empty. This might be normal, or might indicate an unexpected format."
            )
            continue

        if 'OxySoft export of:' in row:
            found_oxysoft_export_line = True
            try:
                metadata['Original file'] = row[1]
            except IndexError:
                logger.warning(
                    f"Line {i} indicates 'OxySoft export of:' but does not have the expected second column. Row content: {row}"
                )
                metadata['Original file'] = "Unknown"
        else:
            if len(row) >= 2 and ":" in row[0]:
                key = row[0].split(':')[0].strip()
                value = row[1].strip()
                metadata[key] = value
            else:
                logger.debug(f"Line {i} doesn't match expected metadata format. Row: {row}")

    if not found_oxysoft_export_line:
        logger.warning(
            f"Didn't find an 'OxySoft export of:' line in the first 7 lines of '{file_path}'. Metadata may be incomplete."
        )

    metadata['Export file'] = file_path
    return metadata


def _read_data(rows: list, file_path: str) -> pd.DataFrame:
    """
    Internal helper function to parse the data portion of the Oxysoft .txt file.
    Returns a DataFrame with columns for O2Hb, HHb, Sample number, Event, etc.
    """
    rows_copy = rows.copy()
    start = None
    end = None
    sample_rate = None

    for idx, row in enumerate(rows_copy):
        if "Datafile sample rate:" in row:
            try:
                sample_rate = int(float(row[1]))
            except (ValueError, IndexError):
                logger.error(f"Could not parse sample rate from row {idx}. Row content: {row}")
                sample_rate = None
        elif "(Sample number)" in row:
            start = idx
        elif "(Event)" in row:
            end = idx
            break

    if start is None or end is None or sample_rate is None:
        msg = (
            f"Could not find required markers in '{file_path}'.\n"
            f"start={start}, end={end}, sample_rate={sample_rate}"
        )
        logger.error(msg)
        raise ValueError(msg)

    # The column labels are in rows from 'start' to 'end'
    col_label_rows = rows_copy[start: end + 1]
    try:
        col_labels = [r[1] for r in col_label_rows]
    except IndexError as e:
        logger.error(
            f"Column label rows do not have the expected structure in file '{file_path}'. Row content: {col_label_rows}"
        )
        raise

    # Clean up column labels: For signals, remove the trailing part (e.g., "O2Hb(1)" -> "O2Hb")
    for idx, label in enumerate(col_labels):
        if "O2Hb" in label or "HHb" in label:
            new_label = label.split('(')[0].strip()
            col_labels[idx] = new_label
        elif "(Sample number)" in label or "(Event)" in label:
            parts = label.split('(')
            if len(parts) == 2:
                new_label = parts[1].split(')')[0]
                col_labels[idx] = new_label
            else:
                logger.warning(f"Unexpected format for column label '{label}'. Leaving it as is.")
        else:
            logger.warning(f"Unexpected value in column labels: '{label}'.")

    # Data section starts after end + 4 lines
    data_rows = rows_copy[end + 4:]
    if not data_rows:
        msg = (
            f"No data rows found after line {end + 4} in file '{file_path}'. "
            "Check if the file is truncated or incorrectly formatted."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Remove the last line if it is empty
    if len(data_rows[-1]) == 1 and data_rows[-1][0] == '':
        data_rows = data_rows[:-1]

    # Clean each row to have the expected number of columns
    clean_data_rows = []
    for idx, row in enumerate(data_rows):
        if len(row) == len(col_labels) + 1:
            if row[-1] == '':
                row.pop()
            else:
                logger.warning(
                    f"Row {idx} has {len(row)} items (1 too many), but the extra item is not empty. Row content: {row}"
                )
                row.pop()
        elif len(row) != len(col_labels):
            logger.error(
                f"Row {idx} has {len(row)} columns, expected {len(col_labels)}. Row content: {row}"
            )
            continue
        clean_data_rows.append(row)

    df = pd.DataFrame(data=clean_data_rows, columns=col_labels)

    # Drop the first 'sample_rate' rows (assumed to be the first second of recording)
    if sample_rate and sample_rate > 0 and len(df) > sample_rate:
        df.drop(df.index[range(sample_rate)], inplace=True)
        logger.debug(
            f"Dropped the first {sample_rate} rows of data in '{file_path}' to remove 1 second of recording."
        )
    else:
        logger.warning(
            f"Sample rate is {sample_rate}, or the dataframe is too short. Not dropping the first second."
        )

    # Convert columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='ignore')

    # Replace empty strings with NaN in the 'Event' column if it exists
    if 'Event' in df.columns:
        mask_empty = (df['Event'] == '')
        if mask_empty.any():
            df.loc[mask_empty, 'Event'] = np.nan
            logger.debug(f"Replaced empty strings with NaN in 'Event' column for file '{file_path}'.")
    else:
        logger.warning(f"No 'Event' column found in file '{file_path}'. This may be normal or unexpected.")

    return df


def _reassign_channels(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Reassign column names into pairs: CH0 HbO, CH0 HbR, CH1 HbO, CH1 HbR, etc.
    If the file has an odd number of data columns (after removing Sample number/Event),
    logs a warning and renames the available columns.
    """
    cols = list(df.columns)

    sample_col = None
    event_col = None
    if "Sample number" in cols:
        sample_col = "Sample number"
        cols.remove("Sample number")
    if "Event" in cols:
        event_col = "Event"
        cols.remove("Event")

    if len(cols) == 0:
        logger.warning(f"No data columns to rename in '{file_path}'.")
        return df  # nothing to do

    if len(cols) % 2 != 0:
        logger.warning(
            f"Data columns in '{file_path}' are not an even number ({len(cols)}). "
            "Cannot reliably split into HbO/HbR pairs. We'll do the best we can."
        )

    num_channels = len(cols) // 2  # integer division
    new_data_cols = []
    # Use zero-based indexing: CH0, CH1, etc.
    for i in range(num_channels):
        new_data_cols.append(f"CH{i} HbO")
        new_data_cols.append(f"CH{i} HbR")

    extra_cols = cols[2 * num_channels:]  # any leftover columns

    new_cols_order = []
    if sample_col is not None:
        new_cols_order.append(sample_col)
    new_cols_order.extend(new_data_cols)
    new_cols_order.extend(extra_cols)
    if event_col is not None:
        new_cols_order.append(event_col)

    if len(new_cols_order) != len(df.columns):
        logger.warning(
            f"New column order has {len(new_cols_order)} columns but df has {len(df.columns)}. "
            f"Skipping renaming for '{file_path}'."
        )
        return df

    # Force assign the new column names
    df.columns = new_cols_order
    logger.debug(f"Reassigned columns to: {new_cols_order}")
    return df
