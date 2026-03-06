"""
Assembly functions for MD ImageXpress HCS.ai microscope data.

This module will contain functions to assemble filenames and load data
from the MD ImageXpress HCS.ai microscope.
"""

import pandas as pd
import xarray as xr


def create_well_filename_structure_MD(
    well_files: pd.DataFrame,
    only_2D: bool = False,
) -> xr.DataArray:
    """
    Assemble MD tiff-filenames of one well in an xarray.

    Args:
        well_files (pd.DataFrame): DataFrame containing the filenames and metadata
            about well, field, times, etc.
            Needs to only contain the files of one well!
        only_2D (bool, optional): If True, only the existing 2D data (i.e.
            projections) are used. The 'plane' axis will still be created, but
            with length 1. Defaults to False.

    Returns:
        xr.DataArray: An xarray containing the filenames in the correct
            structure.
    """
    # TODO: Implement for HCS.ai data structure
    raise NotImplementedError(
        "This function needs to be implemented for the HCS.ai microscope."
    )


def lazy_load_well(
    fns_xr: xr.DataArray,
) -> xr.Dataset:
    """
    Load a well as an xarray dataset with dask arrays.

    Args:
        fns_xr (xr.DataArray): An xarray DataArray containing the filenames
            in the correct structure (field, time, channel, plane).

    Returns:
        xr.Dataset: An xarray Dataset containing the images and stage positions.
    """
    # TODO: Implement for HCS.ai data structure
    raise NotImplementedError(
        "This function needs to be implemented for the HCS.ai microscope."
    )
