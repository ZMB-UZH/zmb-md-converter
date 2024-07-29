import re

import numpy as np
import pandas as pd
import xarray as xr


def create_filename_structure_MD(
    files: pd.DataFrame,
    only_2D: bool = False,
) -> xr.DataArray:
    """
    Assemble MD tiff-filenames in an xarray.

    The array will have the ordering (well,field,time,channel,plane).
    This allows us to later easily map over the filenames to create a
    dask-array of the images.

    Args:
        files (pd.DataFrame): DataFrame containing the filenames and metadata
            about well, field, times, etc.
        only_2D (bool, optional): If True, only the existing 2D data (i.e.
            projections) are used. The 'plane' axis will still be created, but
            with length 1. Defaults to False.

    Returns:
        xr.DataArray: An xarray containing the filenames in the correct order.
    """
    if all(files.z.unique() == [None]):
        only_2D = True

    if only_2D:
        files = files[files.z.isnull()].copy()
        if files.empty:
            raise ValueError("No projections found.")
        files.z = "0"
    else:
        files = files[files.z.notnull()].copy()

    wells = sorted(files["well"].unique())
    fields = sorted(
        files["field"].unique(), key=lambda s: int(re.findall(r"(\d+)", s)[0])
    )
    times = sorted(files["time_point"].unique(), key=int)
    channels = sorted(
        files["channel"].unique(), key=lambda s: int(re.findall(r"(\d+)", s)[0])
    )
    planes = sorted(files["z"].unique(), key=int)

    # Create an empty np array to store the filenames in the correct structure
    fn_dtype = f"<U{max([len(fn) for fn in files['path']])}"
    fns_np = np.zeros(
        (len(wells), len(fields), len(times), len(channels), len(planes)),
        dtype=fn_dtype,
    )

    # Store fns in correct position
    for w, well in enumerate(wells):
        well_files = files[files["well"] == well]
        for s, field in enumerate(fields):
            field_files = well_files[well_files["field"] == field]
            for t, time in enumerate(times):
                time_files = field_files[field_files["time_point"] == time]
                for c, channel in enumerate(channels):
                    channel_files = time_files[time_files["channel"] == channel]
                    for z, plane in enumerate(planes):
                        plane_files = channel_files[channel_files["z"] == plane]
                        if len(plane_files) == 1:
                            fns_np[w, s, t, c, z] = plane_files["path"].values[0]
                        elif len(plane_files) > 1:
                            raise RuntimeError(
                                f"Multiple files found for well {well}, field {field}, "
                                f"time {time}, channel {channel}, plane {plane}."
                            )

    # create xarray
    fns_xr = xr.DataArray(
        fns_np,
        dims=("well", "field", "time", "channel", "plane"),
        coords={
            "well": wells,
            "field": fields,
            "time": times,
            "channel": channels,
            "plane": planes,
        },
    )

    return fns_xr
