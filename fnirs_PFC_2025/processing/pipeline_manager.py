# pipeline_manager.py
import os
import pandas as pd
from fnirs_PFC_2025.processing.file_processor import FileProcessor
from fnirs_PFC_2025.processing.stats_collector import StatsCollector
from fnirs_PFC_2025.read.loaders import read_txt_file
import logging

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages the complete fNIRS processing pipeline with two-pass processing.
    """

    def __init__(self, fs=50.0, sci_threshold=0.6):
        """
        Initialize with processing parameters.

        Args:
            fs: Sampling frequency (Hz)
            sci_threshold: Threshold for SCI calculation
        """
        self.fs = fs
        self.sci_threshold = sci_threshold
        self.file_processor = FileProcessor(fs=fs, sci_threshold=sci_threshold)
        self.stats_collector = StatsCollector(fs=fs)

    def find_input_files(self, input_base_dir):
        """
        Find all .txt files in directory tree.

        Args:
            input_base_dir: Root directory to search

        Returns:
            List of absolute file paths
        """
        txt_files = []
        for root, _, files in os.walk(input_base_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        return sorted(txt_files)

    def run_processing_pass(self, file_paths, output_base_dir, input_base_dir,
                            subject_y_limits=None):
        """
        Execute a processing pass on all files.

        Args:
            file_paths: List of file paths to process
            output_base_dir: Root output directory
            input_base_dir: Root input directory
            subject_y_limits: Optional y-limits for consistent plotting

        Returns:
            Tuple of (processed_files, skipped_files)
        """
        processed = []
        skipped = []

        for file_path in file_paths:
            try:
                logger.info(f"Processing: {file_path}")
                result = self.file_processor.process_file(
                    file_path=file_path,
                    output_base_dir=output_base_dir,
                    input_base_dir=input_base_dir,
                    subject_y_limits=subject_y_limits,
                    read_file_func=read_txt_file  # Pass the loader function
                )
                if result is not None:
                    processed.append(file_path)
                else:
                    skipped.append(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                skipped.append(file_path)

        return processed, skipped

    def generate_reports(self, stats_df, output_dir):
        """Generate all output reports and summaries"""
        # Save combined statistics
        stats_path = os.path.join(output_dir, 'all_subjects_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Saved combined statistics to: {stats_path}")

        # Create condition-specific summaries
        self.stats_collector.create_summary_sheets(stats_df, output_dir)

    def run_pipeline(self, input_base_dir, output_base_dir):
        """
        Execute complete two-pass processing pipeline.

        Args:
            input_base_dir: Root input directory
            output_base_dir: Root output directory

        Returns:
            Combined statistics DataFrame
        """
        # Setup output directory
        os.makedirs(output_base_dir, exist_ok=True)

        # Find all input files
        txt_files = self.find_input_files(input_base_dir)
        logger.info(f"Found {len(txt_files)} .txt files to process")

        # First pass - initial processing
        logger.info("--- First pass: Initial processing ---")
        processed_files, _ = self.run_processing_pass(
            txt_files, output_base_dir, input_base_dir)

        # Calculate consistent y-limits per subject
        logger.info("--- Calculating consistent y-limits ---")
        subject_y_limits = self.stats_collector.calculate_subject_y_limits(
            processed_files, output_base_dir, input_base_dir)

        # Second pass - processing with consistent y-limits
        logger.info("--- Second pass: Processing with y-limits ---")
        final_processed, skipped = self.run_processing_pass(
            txt_files, output_base_dir, input_base_dir, subject_y_limits)

        # Generate statistics and reports
        logger.info("--- Generating reports ---")
        stats_df = self.stats_collector.collect_statistics(
            final_processed, output_base_dir, input_base_dir)
        self.generate_reports(stats_df, output_base_dir)

        # Log final summary
        logger.info("\n--- Processing Summary ---")
        logger.info(f"Total files: {len(txt_files)}")
        logger.info(f"Successfully processed: {len(final_processed)}")
        logger.info(f"Skipped: {len(skipped)}")

        return stats_df