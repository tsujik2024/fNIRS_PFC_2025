import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Optional, Dict, Tuple, Callable, List
import logging

# Import processing steps
from fnirs_PFC_2025.preprocessing.fir_filter import fir_filter
from fnirs_PFC_2025.preprocessing.SCI import calc_sci
from fnirs_PFC_2025.preprocessing.short_channel_regression import scr_regression
from fnirs_PFC_2025.preprocessing.tddr import tddr
from fnirs_PFC_2025.preprocessing.baseline_correction import baseline_subtraction
from fnirs_PFC_2025.preprocessing.average_channels import average_channels

# Import plotting functions
from fnirs_PFC_2025.viz.plots import plot_channels_separately, plot_overall_signals

logger = logging.getLogger(__name__)
plt.ioff()  # Non-interactive backend


class FileProcessor:
    """Handles processing of individual fNIRS files through the complete pipeline."""

    def __init__(self, fs: float = 50.0, sci_threshold: float = 0.6):
        """
        Initialize processor with parameters.

        Args:
            fs: Sampling frequency in Hz
            sci_threshold: Threshold for SCI calculation
        """
        self.fs = fs
        self.sci_threshold = sci_threshold
        logger.info(f"Initialized FileProcessor (fs={fs}, SCI threshold={sci_threshold})")

    def process_file(self, file_path: str, output_base_dir: str,
                     input_base_dir: str, subject_y_limits: Optional[Dict] = None,
                     read_file_func: Callable = None) -> Optional[pd.DataFrame]:
        """
        Process single file through complete pipeline.

        Args:
            file_path: Path to input file
            output_base_dir: Base output directory
            input_base_dir: Base input directory
            subject_y_limits: Dictionary of y-limits for plotting
            read_file_func: Function to use for reading files

        Returns:
            Processed DataFrame or None if failed
        """
        try:
            # Setup output directory
            output_dir = self._create_output_dir(output_base_dir, input_base_dir, file_path)
            file_basename = os.path.basename(file_path)
            subject = self._extract_subject(file_path)

            # Get plotting limits
            raw_limits = self._get_plotting_limits(subject, subject_y_limits)

            # 1) Load and prepare data
            logger.info(f"Processing {file_path}")
            data_dict = read_file_func(file_path)
            data = self._prepare_data(data_dict['data'])

            # 2) Plot raw data only (skip SCI plots)
            self._plot_raw_data(data, output_dir, file_basename, subject, raw_limits)

            # 3-6) Process through pipeline (SCI, SCR, TDDR, baseline correction)
            processed_data = self._process_pipeline_stages(data, output_dir, file_basename)

            # 7) Generate final outputs
            return self._finalize_outputs(processed_data, output_dir, file_basename, subject)

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}", exc_info=True)
            return None

    # Private helper methods ---------------------------------------------------

    @staticmethod
    def _create_output_dir(output_base: str, input_base: str, file_path: str) -> str:
        """Create output directory mirroring input structure."""
        relative_path = os.path.relpath(os.path.dirname(file_path), start=input_base)
        output_dir = os.path.join(output_base, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def _extract_subject(file_path: str) -> str:
        """Extract subject ID from file path."""
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if "OHSU_Turn" in part or any(x in part for x in ["Subject", "subj", "sub-"]):
                return part
        return "Unknown"

    @staticmethod
    def _get_plotting_limits(subject: str, subject_y_limits: Optional[Dict]) -> Tuple:
        """Get plotting limits for raw data."""
        if not subject_y_limits or subject not in subject_y_limits:
            return None
        return (subject_y_limits[subject]['raw_min'], subject_y_limits[subject]['raw_max'])

    @staticmethod
    def _prepare_data(raw_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare raw data DataFrame."""
        data = raw_data.copy()
        if "Sample number" not in data.columns:
            data.insert(0, "Sample number", np.arange(len(data)))
        return data

    def _plot_raw_data(self, data: pd.DataFrame, output_dir: str,
                       file_basename: str, subject: str,
                       global_ylim: Optional[Tuple[float, float]]) -> None:
        """Plot raw data for both ST and DT conditions separately."""
        o2hb_cols = [col for col in data.columns if ('O2Hb' in col or 'HbO' in col)]
        hhb_cols = [col for col in data.columns if ('HHb' in col or 'HbR' in col)]
        combined_cols = o2hb_cols + hhb_cols

        if not combined_cols:
            return

        # Determine condition (ST/DT)
        condition = "ST" if "ST" in file_basename else "DT" if "DT" in file_basename else "UNKNOWN"
        condition_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        # Plot individual channels
        plt.figure(figsize=(15, 10))
        fig, axes, ylim = plot_channels_separately(
            data[combined_cols],
            fs=self.fs,
            title=f"{file_basename} - Raw Data",
            subject=subject,
            condition=condition,
            y_lim=global_ylim
        )
        self._save_figure(condition_dir, f"raw_individual_channels_{condition}.png",
                          f"{file_basename} - Raw Data")

        # Plot overall signals
        if o2hb_cols and hhb_cols:
            avg_o2hb = data[o2hb_cols].mean(axis=1)
            avg_hhb = data[hhb_cols].mean(axis=1)
            overall_df = pd.DataFrame({"grand oxy": avg_o2hb, "grand deoxy": avg_hhb})

            plt.figure(figsize=(15, 6))
            fig, ylim = plot_overall_signals(
                overall_df,
                fs=self.fs,
                title=f"{file_basename} - Raw Overall",
                subject=subject,
                condition=condition,
                y_lim=global_ylim
            )
            self._save_figure(condition_dir, f"raw_overall_{condition}.png",
                              f"{file_basename} - Raw Overall")

    def _process_pipeline_stages(self, data: pd.DataFrame,
                                 output_dir: str, file_basename: str) -> pd.DataFrame:
        """Process data through all pipeline stages (SCI, SCR, TDDR, baseline)."""
        # Calculate SCI (but don't plot)
        o2hb_cols = [col for col in data.columns if ('O2Hb' in col or 'HbO' in col)]
        hhb_cols = [col for col in data.columns if ('HHb' in col or 'HbR' in col)]
        filtered = fir_filter(data, order=1000, Wn=[0.01, 0.1], fs=int(self.fs))
        self._calculate_sci(filtered, o2hb_cols, hhb_cols, output_dir, file_basename)

        # Short Channel Regression
        scr_data = self._apply_scr(data)

        # TDDR
        tddr_data = self._apply_tddr(scr_data)

        # Baseline Correction
        return self._apply_baseline_correction(tddr_data)

    def _calculate_sci(self, filtered_data: pd.DataFrame,
                       o2hb_cols: List[str], hhb_cols: List[str],
                       output_dir: str, file_basename: str) -> None:
        """Calculate SCI and log bad channels."""
        flagged = []
        for o2hb_col in o2hb_cols:
            ch_id = o2hb_col.split()[0]
            hhb_col = next((c for c in hhb_cols if c.startswith(ch_id)), None)

            if hhb_col:
                sci_val = calc_sci(
                    filtered_data[o2hb_col].to_numpy(dtype=np.float64),
                    filtered_data[hhb_col].to_numpy(dtype=np.float64),
                    fs=self.fs, apply_filter=False
                )
                if sci_val < self.sci_threshold:
                    flagged.append((ch_id, sci_val))

        if flagged:
            sci_log = os.path.join(output_dir, f"{os.path.splitext(file_basename)[0]}_bad_SCI_channels.txt")
            with open(sci_log, 'w') as f:
                f.write("Channel\tSCI\n")
                for ch_id, sci_val in flagged:
                    f.write(f"{ch_id}\t{sci_val:.3f}\n")

    def _apply_scr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Short Channel Regression."""
        try:
            short_ch_indices = [7, 8]  # Update with your actual indices
            long_ch_indices = [i for i in range(10) if i not in short_ch_indices]

            short_cols = [col for col in data.columns
                          if any(f"CH{i}" in col for i in short_ch_indices)]
            long_cols = [col for col in data.columns
                         if any(f"CH{i}" in col for i in long_ch_indices)]

            if short_cols and long_cols:
                scr_data = scr_regression(data[long_cols], data[short_cols])
                return pd.concat([scr_data, data.drop(columns=long_cols + short_cols)], axis=1)
            return data
        except Exception as e:
            logger.warning(f"SCR failed: {str(e)}")
            return data

    def _apply_tddr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply TDDR correction."""
        try:
            return tddr(data, sample_rate=self.fs)
        except Exception as e:
            logger.warning(f"TDDR failed: {str(e)}")
            return data

    def _apply_baseline_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply baseline correction."""
        try:
            total_samples = len(data)
            events = pd.DataFrame({
                'Sample number': [
                    int(total_samples - 150 * self.fs),
                    int(total_samples - 130 * self.fs),
                    int(total_samples - 10 * self.fs)
                ],
                'Event': ['S1', 'S2', 'S3']
            })
            return baseline_subtraction(data, events)
        except Exception as e:
            logger.warning(f"Baseline correction failed: {str(e)}")
            return data

    def _finalize_outputs(self, data: pd.DataFrame, output_dir: str,
                          file_basename: str, subject: str) -> pd.DataFrame:
        """Generate final processed outputs with standardized scaling."""
        try:
            # 1) Apply channel averaging
            averaged = average_channels(data)

            if 'grand oxy' not in averaged.columns or 'grand deoxy' not in averaged.columns:
                logger.warning("Missing required columns for final processing")
                return averaged

            # 2) Add time and smoothing
            averaged['Time (s)'] = averaged.get('Sample number', averaged.index) / self.fs
            window = int(self.fs) if int(self.fs) % 2 == 1 else int(self.fs) + 1
            averaged['smoothed oxy'] = savgol_filter(averaged['grand oxy'], window, 2)
            averaged['smoothed deoxy'] = savgol_filter(averaged['grand deoxy'], window, 2)

            # 3) Determine condition (ST/DT)
            condition = "ST" if "ST" in file_basename else "DT" if "DT" in file_basename else "UNKNOWN"
            condition_dir = os.path.join(output_dir, condition)
            os.makedirs(condition_dir, exist_ok=True)

            # 4) Create final processed plot (non-smoothed)
            self._create_final_plot(averaged, condition_dir, file_basename, condition,
                                    columns=['grand oxy', 'grand deoxy'],
                                    prefix="FINAL_processed",
                                    title="Fully Processed")

            # 5) Create smoothed plot
            self._create_final_plot(averaged, condition_dir, file_basename, condition,
                                    columns=['smoothed oxy', 'smoothed deoxy'],
                                    prefix="FINAL_smoothed",
                                    title="Smoothed Signals")

            # 6) Save data
            output_file = os.path.join(condition_dir, f"{file_basename}_FULLY_PROCESSED.csv")
            averaged.to_csv(output_file, index=False)
            logger.info(f"Saved fully processed data to {output_file}")

            return averaged

        except Exception as e:
            logger.error(f"Final output generation failed: {str(e)}", exc_info=True)
            return data

    def _create_final_plot(self, data: pd.DataFrame, output_dir: str,
                           file_basename: str, condition: str,
                           columns: List[str], prefix: str,
                           title: str) -> None:
        """Create standardized final plots with automatic scaling."""
        try:
            plt.figure(figsize=(15, 6))

            # Prepare plot data with consistent column names
            plot_data = data[columns + ['Time (s)']].rename(columns={
                columns[0]: "grand oxy",
                columns[1]: "grand deoxy"
            })

            # Generate plot with auto-scaling
            plot_overall_signals(
                plot_data,
                fs=self.fs,
                title=f"{file_basename} - {title}",
                subject=self._extract_subject(file_basename),
                condition=condition,
                y_lim=None  # Auto-scale
            )

            # Save with consistent naming
            output_path = os.path.join(output_dir, f"{prefix}_{condition}.png")
            self._save_figure(output_dir, os.path.basename(output_path),
                              f"{file_basename} - {title}")

            logger.debug(f"Saved {title} plot to {output_path}")

        except Exception as e:
            logger.error(f"Failed to create {title} plot for {file_basename}: {str(e)}")
            raise

    @staticmethod
    def _save_figure(output_dir: str, filename: str, title: str) -> None:
        """Save matplotlib figure with proper cleanup."""
        try:
            plt.title(title)
            plt.tight_layout()
            path = os.path.join(output_dir, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.debug(f"Saved figure to {path}")
        except Exception as e:
            logger.error(f"Failed to save figure {filename}: {str(e)}")
            plt.close()
            raise