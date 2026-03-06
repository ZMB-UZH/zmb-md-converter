"""
Parsing functions for MD ImageXpress HCS.ai microscope data.

This module will contain functions to parse the file structure and metadata
from the MD ImageXpress HCS.ai microscope.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def parse_MD_plate_folder(root_dir: Union[Path, str]) -> Optional[pd.DataFrame]:
    """
    Parse an MD-plate folder for HCS.ai data structure.

    Args:
        root_dir (Union[Path, str]): The root directory of the MD-plate.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the parsed data,
            or None if no valid data structure is found.
    """
    # TODO: Implement for HCS.ai data structure
    raise NotImplementedError(
        "This function needs to be implemented for the HCS.ai microscope."
    )


def parse_MD_tz_folder(root_dir: Union[Path, str]) -> Optional[pd.DataFrame]:
    """
    Parse an MD time-z folder for HCS.ai data structure.

    Args:
        root_dir (Union[Path, str]): The root directory of the MD-plate.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the parsed data,
            or None if no valid data structure is found.
    """
    # TODO: Implement for HCS.ai data structure
    raise NotImplementedError(
        "This function needs to be implemented for the HCS.ai microscope."
    )
