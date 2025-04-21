"""
Data loading and input/output operations for fNIRS data.
"""

from .loaders import (
    read_txt_file,           # Main public function
    _reassign_channels,      # Internal but needed by other modules
    _read_metadata,          # Internal but needed
    _read_data               # Internal but needed
)

# Explicit exports
__all__ = [
    'read_txt_file'          # Only expose this to public API
]

# Internal imports for cross-module use
__internals__ = [
    '_reassign_channels',
    '_read_metadata',
    '_read_data'
]