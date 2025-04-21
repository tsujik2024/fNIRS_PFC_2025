"""
Processing module for fNIRS data analysis.
Contains classes and functions for file processing and statistics collection.
"""

from .file_processor import FileProcessor
from .stats_collector import StatsCollector
from .pipeline_manager import PipelineManager
from .batch_processor import BatchProcessor
__all__ = ['FileProcessor', 'StatsCollector', 'PipelineManager', 'BatchProcessor']