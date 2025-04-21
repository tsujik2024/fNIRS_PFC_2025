#!/usr/bin/env python3
"""
fNIRS Processing Pipeline - Command Line Interface
"""

import argparse
import logging
from fnirs_PFC_2025.processing.batch_processor import BatchProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Process fNIRS data through the complete pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Directory containing .txt fNIRS files")
    parser.add_argument("output_dir", help="Directory for processed outputs")
    parser.add_argument("--fs", type=float, default=50.0, help="Sampling rate in Hz")
    parser.add_argument("--sci_thresh", type=float, default=0.7,
                        help="Scalp Coupling Index threshold")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("fnirs_processing.log"),
            logging.StreamHandler()
        ]
    )

    try:
        processor = BatchProcessor(fs=args.fs, sci_threshold=args.sci_thresh)
        results = processor.run_two_pass_processing(args.input_dir, args.output_dir)

        print("\nProcessing Complete!")
        print(f"Successfully processed {len(results['processed_files'])}/{results['total_files']} files")
        print(f"Statistics saved to: {args.output_dir}/all_subjects_statistics.csv")

    except Exception as e:
        logging.critical(f"Pipeline failed: {str(e)}", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()