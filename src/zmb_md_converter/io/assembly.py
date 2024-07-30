import re

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import xarray as xr
from numpy.typing import ArrayLike

from zmb_md_converter.io.MetaSeriesTiff import (
    load_metaseries_tiff,
    load_metaseries_tiff_metadata,
)


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
        xr.DataArray: An xarray containing the filenames in the correct
            structure.
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


def _check_if_channel_contains_stack(fns_xr: xr.DataArray, c: int) -> bool:
    fns_sel = fns_xr.data[0, 0, 0, c]

    if "" in fns_sel:
        return False

    if len(fns_sel) < 2:
        return False

    # for MetaXpress exported data, projection-only files are copied across the entire
    # stack, but they lack the "Z Step" metadata
    metadata = load_metaseries_tiff_metadata(fns_sel[0])
    if "Z Step" not in metadata.keys():
        return False

    # for MetaXpress exported data, single plane files are copied across the entire
    # stack, but they all have "Z Step" = 1 in the metadata -> check 2nd file in stack
    metadata = load_metaseries_tiff_metadata(fns_sel[1])
    if metadata["Z Step"] == 1:
        return False

    return True


def _check_if_channel_contains_timeseries(fns_xr: xr.DataArray, c: int) -> bool:
    fns_sel = fns_xr.data[0, 0, :, c, 0]

    if "" in fns_sel:
        return False

    if len(fns_sel) < 2:
        return False

    # for MetaXpress exported data, sparse timepoint files are copied across the entire
    # timeseries, but they all have "Timepoint" = 1 in the metadata -> check 2nd file in
    # stack
    metadata = load_metaseries_tiff_metadata(fns_sel[1])
    if metadata["Timepoint"] == 1:
        return False

    return True


def lazy_load_plate_as_xr(fns_xr: xr.DataArray) -> xr.DataArray:
    """
    Lazily load the images from the filenames in the xarray into a new xarray.

    Args:
        fns_xr (xr.DataArray): An xarray containing the filenames in the
            correct structure.

    Returns:
        xr.DataArray: An xarray with shape WFTCZYX containing the images. The
            data are loaded as a dask array.
    """
    # Get the image dimensions and some metadata from the first image
    fns_flat = fns_xr.data.flatten()
    fn_meta = fns_flat[fns_flat != ""][0]
    image, metadata = load_metaseries_tiff(fn_meta)
    x_dim = metadata["pixel-size-x"]
    y_dim = metadata["pixel-size-y"]
    dx = metadata["spatial-calibration-x"]
    dy = metadata["spatial-calibration-x"]
    dtype = image.dtype

    # Get approximate dz from first stack
    dz = 0
    if fns_xr.shape[-1] > 1:
        for c in range(fns_xr.shape[3]):
            if _check_if_channel_contains_stack(fns_xr, c):
                z_positions = []
                for z in range(fns_xr.shape[-1]):
                    metadata = load_metaseries_tiff_metadata(
                        fns_xr.values[0, 0, 0, c, z]
                    )
                    z_positions.append(metadata["stage-position-z"])
                z_positions_array = np.array(z_positions)
                dz = np.mean(z_positions_array[1:] - z_positions_array[:-1])
                dz = round(dz, 3)
                break
        if dz == 0:
            raise ValueError("More than one plane found, but unable to determine dz.")

    # Get approximate dt from timeseries
    dt = 0
    if fns_xr.shape[2] > 1:
        for c in range(fns_xr.shape[3]):
            if _check_if_channel_contains_timeseries(fns_xr, c):
                timepoints = []
                for t in range(fns_xr.shape[2]):
                    metadata = load_metaseries_tiff_metadata(
                        fns_xr.values[0, 0, t, c, 0]
                    )
                    timepoints.append(metadata["acquisition-time-local"])
                timepoints_array = np.array(timepoints)
                dt = np.mean(
                    timepoints_array[1:] - timepoints_array[:-1]
                ).total_seconds()
                dt = round(dt, 3)
                break
        if dt == 0:
            raise ValueError(
                "More than one timepoint found, but unable to determine dt."
            )

    # function to read images from filenames
    def _read_images(x: ArrayLike, ny: int, nx: int, im_dtype: type) -> ArrayLike:
        images = np.zeros((*x.shape, ny, nx), dtype=im_dtype)
        for i in np.ndindex(x.shape):
            filename = x[i]
            if filename != "":
                images[i] = tifffile.imread(filename)
        return images

    # create dask-array for images by mapping _read_images over fns_xr
    fns_shape = fns_xr.shape
    images_da = da.map_blocks(
        _read_images,
        da.from_array(fns_xr.data, chunks=(1,) * len(fns_shape)),
        chunks=da.core.normalize_chunks(
            (1,) * len(fns_shape) + (x_dim, y_dim), (*fns_shape, x_dim, y_dim)
        ),
        new_axis=list(range(len(fns_shape), len(fns_shape) + 2)),
        meta=np.array([], dtype=dtype),
        # function kwargs
        ny=y_dim,
        nx=x_dim,
        im_dtype=dtype,
    )
    images_xr = xr.DataArray(
        images_da,
        name="images",
        dims=(*fns_xr.dims, "y", "x"),
        coords=fns_xr.coords,
    )
    images_xr.attrs = {
        "dt": dt,
        "dz": dz,
        "dy": dy,
        "dx": dx,
    }

    return images_xr
