import os
import argparse
import logging
from typing import List, Optional, Tuple
from fnirs_PFC_2025.processing.file_processor import FileProcessor
from fnirs_PFC_2025.processing.stats_collector import StatsCollector
from fnirs_PFC_2025.read.loaders import read_txt_file
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of multiple fNIRS files."""

    def __init__(self, fs: float = 50.0, sci_threshold: float = 0.6):
        """
        Initialize batch processor.

        Args:
            fs: Sampling frequency in Hz
            sci_threshold: SCI threshold value
        """
        self.fs = fs
        self.sci_threshold = sci_threshold
        self.file_processor = FileProcessor(fs=fs, sci_threshold=sci_threshold)
        self.stats_collector = StatsCollector(fs=fs)
        logger.info("Initialized BatchProcessor")

    def find_input_files(self, input_dir: str) -> List[str]:
        """Recursively find all .txt files in directory."""
        txt_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        return sorted(txt_files)

    def run_two_pass_processing(self, input_dir: str, output_dir: str) -> dict:
        """
        Run complete two-pass processing pipeline.

        Args:
            input_dir: Input directory containing .txt files
            output_dir: Output directory for processed data

        Returns:
            Dictionary containing:
            - stats: Combined statistics DataFrame
            - y_limits: Calculated y-limits
            - processed_files: List of successfully processed files
            - skipped_files: List of skipped files
        """
        os.makedirs(output_dir, exist_ok=True)
        txt_files = self.find_input_files(input_dir)
        logger.info(f"Found {len(txt_files)} input files")

        if not txt_files:
            raise ValueError("No .txt files found in input directory")

        # First pass - initial processing
        logger.info("Starting first processing pass")
        first_pass_files, first_skipped = self._process_files(
            txt_files, input_dir, output_dir
        )

        # Calculate y-limits
        logger.info("Calculating subject y-limits")
        y_limits = self.stats_collector.calculate_subject_y_limits(
            first_pass_files, output_dir, input_dir
        )

        # Second pass - processing with consistent y-limits
        logger.info("Starting second processing pass with y-limits")
        final_files, final_skipped = self._process_files(
            txt_files, input_dir, output_dir, y_limits
        )

        # Generate statistics
        logger.info("Generating statistics")
        stats = self.stats_collector.collect_statistics(
            final_files, output_dir, input_dir
        )

        # Create summaries
        logger.info("Creating summary sheets")
        self.stats_collector.create_summary_sheets(stats, output_dir)

        # Save complete stats
        stats_path = os.path.join(output_dir, 'all_subjects_statistics.csv')
        stats.to_csv(stats_path, index=False)

        return {
            'stats': stats,
            'y_limits': y_limits,
            'processed_files': final_files,
            'skipped_files': final_skipped + first_skipped,
            'total_files': len(txt_files)
        }

    def _process_files(self, file_paths: List[str], input_dir: str, output_dir: str,
                       y_limits: Optional[dict] = None) -> Tuple[List[str], List[str]]:
        """
        Process a batch of files.

        Args:
            file_paths: List of file paths to process
            input_dir: Base input directory
            output_dir: Base output directory
            y_limits: Optional y-limits for plotting

        Returns:
            Tuple of (processed_files, skipped_files)
        """
        processed = []
        skipped = []

        for file_path in file_paths:
            result = self.file_processor.process_file(
                file_path=file_path,
                output_base_dir=output_dir,
                input_base_dir=input_dir,
                subject_y_limits=y_limits,
                read_file_func=read_txt_file
            )
            if result is not None:
                processed.append(file_path)
            else:
                skipped.append(file_path)

        return processed, skipped


def main():
    """Command line interface for batch processing."""
    parser = argparse.ArgumentParser(
        description="fNIRS Batch Processing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Directory containing input .txt files")
    parser.add_argument("output_dir", help="Directory for processed outputs")
    parser.add_argument("--fs", type=float, default=50.0,
                        help="Sampling frequency in Hz")
    parser.add_argument("--sci_thresh", type=float, default=0.7,
                        help="SCI threshold value")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run pipeline
    processor = BatchProcessor(fs=args.fs, sci_threshold=args.sci_thresh)
    results = processor.run_two_pass_processing(args.input_dir, args.output_dir)

    # Print summary
    print("\nProcessing Summary:")
    print(f"Total files: {results['total_files']}")
    print(f"Successfully processed: {len(results['processed_files'])}")
    print(f"Skipped: {len(results['skipped_files'])}")
    print(f"Statistics saved to: {os.path.join(args.output_dir, 'all_subjects_statistics.csv')}")


if __name__ == "__main__":
    main()