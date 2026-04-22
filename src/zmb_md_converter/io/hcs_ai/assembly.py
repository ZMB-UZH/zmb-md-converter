"""
Assembly functions for MD ImageXpress HCS.ai microscope data.

Unlike the HT.ai module, metadata (pixel size, z-step, channel names, stage
positions) is taken directly from the parsed DataFrame (JDCE + CSV), so no
TIFF header reads are required during assembly.
"""

import re

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import xarray as xr
from numpy.typing import ArrayLike


def create_well_filename_structure_MD(
    well_files: pd.DataFrame,
    only_2D: bool = False,
) -> xr.DataArray:
    """
    Assemble MD tiff-filenames of one well in an xarray.

    The array has the ordering (field, time, channel, plane). Stage positions
    and channel metadata extracted during parsing are stored in ``attrs`` so
    that ``lazy_load_well`` does not need to re-read TIFF headers.

    Args:
        well_files (pd.DataFrame): DataFrame produced by
            ``parse_MD_plate_folder`` filtered to a single well.
        only_2D (bool, optional): If True, only projection rows (z=NaN) are
            used and the plane axis has length 1. Defaults to False.

    Returns:
        xr.DataArray: Filename array with dims (field, time, channel, plane).
    """
    if len(well_files["well"].unique()) > 1:
        raise ValueError(
            "The input DataFrame contains files from multiple wells. "
            "Please filter the DataFrame to only contain files from one well."
        )

    if well_files.z.isnull().all():
        only_2D = True
    if only_2D:
        well_files = well_files[well_files.z.isnull()].copy()
        if well_files.empty:
            raise ValueError("No projections found.")
        well_files["z"] = "0"
    else:
        well_files = well_files[well_files.z.notnull()].copy()

    well = well_files["well"].iloc[0]
    fields = sorted(
        well_files["field"].unique(), key=lambda s: int(re.findall(r"(\d+)", s)[0])
    )
    times = sorted(well_files["time_point"].unique(), key=int)
    channels = sorted(
        well_files["channel"].unique(), key=lambda s: int(re.findall(r"(\d+)", s)[0])
    )
    planes = sorted(well_files["z"].unique(), key=int)

    fn_dtype = f"<U{max(len(fn) for fn in well_files['path'])}"
    fns_np = np.zeros(
        (len(fields), len(times), len(channels), len(planes)), dtype=fn_dtype
    )
    # Stage positions: (field, time, channel, plane, 3) → (z, y, x)
    pos_np = np.zeros(
        (len(fields), len(times), len(channels), len(planes), 3), dtype=float
    )

    for s, field in enumerate(fields):
        field_files = well_files[well_files["field"] == field]
        for t, time in enumerate(times):
            time_files = field_files[field_files["time_point"] == time]
            for c, channel in enumerate(channels):
                channel_files = time_files[time_files["channel"] == channel]
                for z, plane in enumerate(planes):
                    plane_files = channel_files[channel_files["z"] == plane]
                    if len(plane_files) == 1:
                        row = plane_files.iloc[0]
                        fns_np[s, t, c, z] = row["path"]
                        pos_np[s, t, c, z] = [
                            row["pos_z"],
                            row["pos_y"],
                            row["pos_x"],
                        ]
                    elif len(plane_files) > 1:
                        raise RuntimeError(
                            f"Multiple files found for well {well}, field {field}, "
                            f"time {time}, channel {channel}, plane {plane}."
                        )

    # Build per-channel metadata from the DataFrame (derived from JDCE)
    channel_metadata = {}
    for channel in channels:
        row = well_files[well_files["channel"] == channel].iloc[0]
        channel_metadata[channel] = {
            "plate_name": row["name"],
            "well_name": well,
            "channel_name": row["channel_name"],
            "dx": row["dx"],
            "dy": row["dy"],
            "dz": row["dz"],
            "spatial_calibration_units": row["spatial_calibration_units"],
            "dt": row["dt"],
            "time_calibration_units": row["time_calibration_units"],
        }

    fns_xr = xr.DataArray(
        fns_np,
        dims=("field", "time", "channel", "plane"),
        coords={
            "field": fields,
            "time": times,
            "channel": channels,
            "plane": planes,
        },
        attrs={
            "plate_name": well_files["name"].values[0],
            "well_name": well,
            "channel_metadata": channel_metadata,
            # shape (field, time, channel, plane, 3), last dim = (z, y, x)
            "stage_positions": pos_np,
        },
    )

    return fns_xr


def _read_images(x: ArrayLike, ny: int, nx: int, im_dtype: type) -> ArrayLike:
    """Read images from an array of filenames."""
    images = np.zeros((*x.shape, ny, nx), dtype=im_dtype)
    for i in np.ndindex(x.shape):
        filename = x[i]
        if filename != "":
            images[i] = tifffile.imread(filename)
    return images


def lazy_load_images(fns_xr: xr.DataArray) -> xr.DataArray:
    """
    Lazily load images from the filename xarray into a dask-backed DataArray.

    Args:
        fns_xr (xr.DataArray): Filename array produced by
            ``create_well_filename_structure_MD``.

    Returns:
        xr.DataArray: Image array with dims (field, time, channel, plane, y, x).
    """
    fns_flat = fns_xr.data.flatten()
    fn_first = fns_flat[fns_flat != ""][0]
    first_image = tifffile.imread(fn_first)
    ny, nx = first_image.shape
    dtype = first_image.dtype

    fns_shape = fns_xr.shape
    images_da = da.map_blocks(
        _read_images,
        da.from_array(fns_xr.data, chunks=(1,) * len(fns_shape)),
        chunks=da.core.normalize_chunks(
            (1,) * len(fns_shape) + (ny, nx), (*fns_shape, ny, nx)
        ),
        new_axis=list(range(len(fns_shape), len(fns_shape) + 2)),
        meta=np.array([], dtype=dtype),
        ny=ny,
        nx=nx,
        im_dtype=dtype,
    )

    images_xr = xr.DataArray(
        images_da,
        name="images",
        dims=(*fns_xr.dims, "y", "x"),
        coords=fns_xr.coords,
        attrs=fns_xr.attrs["channel_metadata"],
    )

    return images_xr


def lazy_load_well(fns_xr: xr.DataArray) -> xr.Dataset:
    """
    Lazily load well data (images & stage positions) into an xarray Dataset.

    Stage positions are taken from the pre-parsed data stored in
    ``fns_xr.attrs`` — no TIFF headers are read.

    Args:
        fns_xr (xr.DataArray): Filename array produced by
            ``create_well_filename_structure_MD``.

    Returns:
        xr.Dataset: Dataset with variables ``images`` and
            ``stage_positions``, plus coordinates ``coords_x``,
            ``coords_y``, ``coords_z``, ``coords_t``.
    """
    images_xr = lazy_load_images(fns_xr)

    # Stage positions are stored as a numpy array in attrs,
    # shape (field, time, channel, plane, 3) with last dim = (z, y, x)
    pos_np = fns_xr.attrs["stage_positions"]
    positions_xr = xr.DataArray(
        da.from_array(pos_np),
        name="stage_positions",
        dims=(*fns_xr.dims, "pos"),
        coords={**fns_xr.coords, "pos": ["pos_z", "pos_y", "pos_x"]},
    )

    # Per-field x, y stage coordinates (time=0, channel=0, plane=0)
    coords_x = xr.DataArray(
        pos_np[:, 0, 0, 0, 2],
        dims="field",
        coords={"field": fns_xr.coords["field"]},
    )
    coords_y = xr.DataArray(
        pos_np[:, 0, 0, 0, 1],
        dims="field",
        coords={"field": fns_xr.coords["field"]},
    )

    # Per-plane z coordinates from actual stage positions (field=0, time=0, channel=0)
    coords_z = xr.DataArray(
        pos_np[0, 0, 0, :, 0],
        dims="plane",
        coords={"plane": fns_xr.coords["plane"]},
    )

    # Time coordinates from dt in channel metadata
    dt = images_xr.attrs[fns_xr.coords["channel"].values[0]]["dt"]
    coords_t = xr.DataArray(
        data=np.array([n * dt for n in range(images_xr.sizes["time"])]),
        dims="time",
        coords={"time": fns_xr.coords["time"]},
    )

    return xr.Dataset(
        {images_xr.name: images_xr, positions_xr.name: positions_xr},
        coords={
            "coords_x": coords_x,
            "coords_y": coords_y,
            "coords_z": coords_z,
            "coords_t": coords_t,
        },
    )
