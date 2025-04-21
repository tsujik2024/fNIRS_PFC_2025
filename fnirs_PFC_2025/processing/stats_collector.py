import os
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from fnirs_PFC_2025.read.loaders import read_txt_file

# Set up logging
logger = logging.getLogger(__name__)


class StatsCollector:
    """
    Collects and compiles statistics from processed fNIRS files, calculates consistent y-axis limits,
    and generates summary reports.
    """

    def __init__(self, fs: float = 50.0):
        """
        Initialize the stats collector.

        Args:
            fs: Sampling frequency in Hz (used for time calculations)
        """
        self.fs = fs
        logger.info("StatsCollector initialized")

    def collect_statistics(self,
                           processed_files: List[str],
                           output_base_dir: str,
                           input_base_dir: str) -> pd.DataFrame:
        """
        Collect statistics from all processed files and create a combined dataframe.

        Args:
            processed_files: List of original file paths that were successfully processed
            output_base_dir: Base directory where processed files are stored
            input_base_dir: Base directory where input files are stored

        Returns:
            DataFrame with combined statistics for all subjects
        """
        all_stats = []
        logger.info(f"Collecting statistics for {len(processed_files)} files")

        for file_path in processed_files:
            try:
                file_basename = os.path.basename(file_path)
                file_name = os.path.splitext(file_basename)[0]

                # Extract metadata from path
                subject, timepoint, condition = self._extract_metadata_from_path(file_path, file_name)

                # Load processed data
                processed_df = self._load_processed_file(file_path, input_base_dir, output_base_dir)
                if processed_df is None:
                    continue

                # Calculate statistics
                stats = self._calculate_file_statistics(processed_df, subject, timepoint, condition)
                all_stats.append(stats)

                logger.debug(f"Added stats for: {file_name} - Subject: {subject}")

            except Exception as e:
                logger.error(f"Error collecting statistics for {file_path}: {str(e)}", exc_info=True)

        return self._create_stats_dataframe(all_stats)

    def calculate_subject_y_limits(self,
                                   processed_files: List[str],
                                   output_base_dir: str,
                                   input_base_dir: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate consistent y-axis limits for each subject across all their files.
        Includes both raw and processed signal ranges for comprehensive limits.

        Args:
            processed_files: List of original file paths that were successfully processed
            output_base_dir: Base directory where processed files are stored
            input_base_dir: Base directory where input files are stored

        Returns:
            Dictionary with consistent y-axis limits per subject in format:
            {
                'subject_id': {
                    'min': float,
                    'max': float,
                    'raw_min': float,
                    'raw_max': float
                }, ...
            }
        """
        subject_y_limits = {}
        logger.info(f"Calculating y-limits for {len(processed_files)} files")

        for file_path in processed_files:
            try:
                # Extract subject from path
                subject = self._extract_subject_from_path(file_path)
                if subject is None:
                    continue

                # Get paths to check (raw and processed)
                raw_path, processed_path = self._get_data_paths(file_path, input_base_dir, output_base_dir)

                # Calculate min/max values
                current_min, current_max = self._calculate_signal_range(raw_path, processed_path)
                if current_min is None or current_max is None:
                    continue

                # Update subject limits
                self._update_subject_limits(subject_y_limits, subject, current_min, current_max)

            except Exception as e:
                logger.error(f"Error calculating y-limits for {file_path}: {str(e)}", exc_info=True)

        return subject_y_limits

    def create_summary_sheets(self,
                              combined_stats_df: pd.DataFrame,
                              output_folder: str) -> None:
        """
        Create summary sheets for single task and dual task conditions.

        Args:
            combined_stats_df: DataFrame containing combined statistics
            output_folder: Folder to save summary sheets
        """
        try:
            # ST condition summary
            summary_ST = self._filter_and_format_summary(combined_stats_df, 'LongWalk_ST')
            summary_ST_file = os.path.join(output_folder, 'summary_ST.csv')
            summary_ST.to_csv(summary_ST_file, index=False)
            logger.info(f"Saved ST summary to {summary_ST_file}")

            # DT condition summary
            summary_DT = self._filter_and_format_summary(combined_stats_df, 'LongWalk_DT')
            summary_DT_file = os.path.join(output_folder, 'summary_DT.csv')
            summary_DT.to_csv(summary_DT_file, index=False)
            logger.info(f"Saved DT summary to {summary_DT_file}")

        except Exception as e:
            logger.error(f"Error creating summary sheets: {str(e)}", exc_info=True)
            raise

    # Helper methods ------------------------------------------------------------

    def _extract_metadata_from_path(self, file_path: str, file_name: str) -> tuple:
        """Extract subject, timepoint, and condition from file path."""
        path_parts = file_path.split(os.sep)

        # Extract subject
        subject = "Unknown"
        for part in path_parts:
            if "OHSU_Turn" in part or any(x in part for x in ["Subject", "subj", "sub-"]):
                subject = part
                break

        # Extract timepoint
        timepoint = "Unknown"
        for part in path_parts:
            if part.lower() in ["baseline", "pre", "post"]:
                timepoint = part
                break

        # Determine condition
        condition = "Unknown"
        if "ST" in file_name or "SingleTask" in file_name:
            condition = "LongWalk_ST"
        elif "DT" in file_name or "DualTask" in file_name:
            condition = "LongWalk_DT"

        return subject, timepoint, condition

    def _extract_subject_from_path(self, file_path: str) -> Optional[str]:
        """Extract subject ID from file path."""
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if "OHSU_Turn" in part or any(x in part for x in ["Subject", "subj", "sub-"]):
                return part
        logger.warning(f"No subject ID found in path: {file_path}")
        return None

    def _load_processed_file(self, file_path, input_base_dir, output_base_dir):
        """Updated to match FileProcessor's output structure"""
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        relative_path = os.path.relpath(os.path.dirname(file_path), start=input_base_dir)

        # Determine condition (ST/DT)
        condition = "ST" if "ST" in file_name else "DT" if "DT" in file_name else "UNKNOWN"

        # Build the correct processed file path
        processed_file = os.path.join(
            output_base_dir,
            relative_path,
            condition,
            f"{file_name}_FULLY_PROCESSED.csv"  # Matches FileProcessor's naming
        )

        if not os.path.exists(processed_file):
            logger.warning(f"Processed file not found: {processed_file}")
            return None

        try:
            processed_df = pd.read_csv(processed_file)
            if 'grand oxy' not in processed_df.columns:
                logger.warning(f"Missing required columns in {processed_file}")
                return None
            return processed_df
        except Exception as e:
            logger.error(f"Error loading {processed_file}: {str(e)}")
            return None

    def _calculate_file_statistics(self,
                                   processed_df: pd.DataFrame,
                                   subject: str,
                                   timepoint: str,
                                   condition: str) -> Dict[str, Union[str, float]]:
        """Calculate statistics for a single file."""
        total_samples = len(processed_df)
        first_half = processed_df.iloc[:total_samples // 2]
        second_half = processed_df.iloc[total_samples // 2:]

        return {
            'Subject': subject,
            'Timepoint': timepoint,
            'Condition': condition,
            'Overall grand oxy Mean': processed_df['grand oxy'].mean(),
            'First Half grand oxy Mean': first_half['grand oxy'].mean(),
            'Second Half grand oxy Mean': second_half['grand oxy'].mean()
        }

    def _create_stats_dataframe(self, all_stats: List[Dict]) -> pd.DataFrame:
        """Convert list of stats dictionaries to DataFrame."""
        if all_stats:
            return pd.DataFrame(all_stats)
        else:
            logger.warning("No statistics collected - returning empty DataFrame")
            return pd.DataFrame(columns=[
                'Subject', 'Timepoint', 'Condition',
                'Overall grand oxy Mean', 'First Half grand oxy Mean',
                'Second Half grand oxy Mean'
            ])

    def _get_data_paths(self,
                        file_path: str,
                        input_base_dir: str,
                        output_base_dir: str) -> tuple:
        """Get paths to raw and processed data files."""
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        relative_path = os.path.relpath(os.path.dirname(file_path), start=input_base_dir)
        output_dir = os.path.join(output_base_dir, relative_path)
        processed_path = os.path.join(output_dir, f"{file_name}_processed.csv")
        return file_path, processed_path

    def _calculate_signal_range(self,
                                raw_path: str,
                                processed_path: str) -> tuple:
        """Calculate min/max signal values from raw and processed data."""
        current_min = float('inf')
        current_max = -float('inf')

        for file_to_check in [raw_path, processed_path]:
            if not os.path.exists(file_to_check):
                continue

            try:
                if file_to_check == raw_path:  # Raw data
                    raw_dict = read_txt_file(raw_path)
                    df = raw_dict["data"].copy()
                    o2hb_cols = [col for col in df.columns if ('O2Hb' in col or 'HbO' in col)]
                    hhb_cols = [col for col in df.columns if ('HHb' in col or 'HbR' in col)]
                    if not o2hb_cols or not hhb_cols:
                        continue
                    all_values = pd.concat([df[o2hb_cols], df[hhb_cols]], axis=1).values
                else:  # Processed data
                    df = pd.read_csv(processed_path)
                    if 'grand oxy' not in df.columns or 'grand deoxy' not in df.columns:
                        continue
                    all_values = df[['grand oxy', 'grand deoxy']].values

                file_min = np.nanmin(all_values)
                file_max = np.nanmax(all_values)
                current_min = min(current_min, file_min)
                current_max = max(current_max, file_max)

            except Exception as e:
                logger.warning(f"Error processing {file_to_check}: {str(e)}")
                continue

        if current_min == float('inf') or current_max == -float('inf'):
            return None, None

        return current_min, current_max

    def _update_subject_limits(self,
                               subject_y_limits: Dict,
                               subject: str,
                               current_min: float,
                               current_max: float) -> None:
        """Update subject limits dictionary with new min/max values."""
        # Add 10% padding
        padding = (current_max - current_min) * 0.1
        current_min -= padding
        current_max += padding

        if subject not in subject_y_limits:
            subject_y_limits[subject] = {
                'min': current_min,
                'max': current_max,
                'raw_min': current_min,
                'raw_max': current_max
            }
        else:
            subject_y_limits[subject]['min'] = min(
                subject_y_limits[subject]['min'], current_min)
            subject_y_limits[subject]['max'] = max(
                subject_y_limits[subject]['max'], current_max)
            subject_y_limits[subject]['raw_min'] = min(
                subject_y_limits[subject]['raw_min'], current_min)
            subject_y_limits[subject]['raw_max'] = max(
                subject_y_limits[subject]['raw_max'], current_max)

    def _filter_and_format_summary(self,
                                   df: pd.DataFrame,
                                   condition: str) -> pd.DataFrame:
        """Filter and format summary for specific condition."""
        filtered = df[df['Condition'] == condition]
        return filtered[[
            'Subject', 'Timepoint',
            'Overall grand oxy Mean',
            'First Half grand oxy Mean',
            'Second Half grand oxy Mean'
        ]]