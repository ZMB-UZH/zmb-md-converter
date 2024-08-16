import re

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import xarray as xr
from numpy.typing import ArrayLike

from zmb_md_converter.io.metaseries import (
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
        attrs={"plate_name": files["name"].values[0]},
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


def _get_z_spacing(fns_xr: xr.DataArray) -> float:
    """Get approximate dz from first stack."""
    # TODO: maybe calculated from lazy_load_stage_positions
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

    return dz


def _get_t_spacing(fns_xr: xr.DataArray) -> float:
    """Get approximate dt from timeseries."""
    # TODO: maybe calculated from lazy_load_stage_positions (need to include time there)
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

    return dt


def _build_channel_metadata(fns_xr: xr.DataArray) -> dict:
    dz = _get_z_spacing(fns_xr)
    dt = _get_t_spacing(fns_xr)

    channel_metadata = {}
    for c, channel in enumerate(fns_xr.channel.values):
        metadata = load_metaseries_tiff_metadata(fns_xr.values[0, 0, 0, c, 0])
        if "Z Projection Method" in metadata:
            name = (
                f"{metadata['Z Projection Method'].replace(' ', '-')}-Projection_"
                f"{metadata['_IllumSetting_']}"
            )
        else:
            name = metadata["_IllumSetting_"]
        spatial_calibration_units = metadata["spatial-calibration-units"]
        if spatial_calibration_units == "um":
            spatial_calibration_units = "µm"
        channel_metadata[channel] = {
            # 'channel_index': int(channel[1:]),  # Note: faim-ipa uses -1
            "plate_name": fns_xr.attrs["plate_name"],
            "channel_name": name,
            "dx": metadata["spatial-calibration-x"],
            "dy": metadata["spatial-calibration-y"],
            "dz": dz,
            "spatial_calibration_units": spatial_calibration_units,
            "dt": dt,
            "time_calibration_units": "s",
            "wavelength": metadata["wavelength"],
            "exposure_time": float(metadata["Exposure Time"].split(" ")[0]),
            "exposure_time_unit": metadata["Exposure Time"].split(" ")[1],
            "objective": metadata["_MagSetting_"],
        }
    return channel_metadata


def _read_stage_positions(x: ArrayLike) -> ArrayLike:
    """Function to read stage positions from filenames."""
    stage_positions = np.zeros((*x.shape, 3), dtype=float)
    for i in np.ndindex(x.shape):
        filename = x[i]
        if filename != "":
            metadata = load_metaseries_tiff_metadata(filename)
            stage_positions[i] = [
                metadata["stage-position-z"],
                metadata["stage-position-y"],
                metadata["stage-position-x"],
            ]
    return stage_positions


def lazy_load_stage_positions(fns_xr: xr.DataArray) -> xr.DataArray:
    """
    Lazily load the stage positions (z,y,x) from the filenames in the xarray.

    Args:
        fns_xr (xr.DataArray): An xarray containing the filenames in the
            correct structure.

    Returns:
        xr.DataArray: An xarray with shape WFTCZ3 containing the stage
            positions. The data are loaded as a dask array.
    """
    # create dask-array for images by mapping _read_stage_positions over fns_xr
    fns_shape = fns_xr.shape
    positions_da = da.map_blocks(
        _read_stage_positions,
        da.from_array(fns_xr.data, chunks=(1,) * len(fns_shape)),
        chunks=da.core.normalize_chunks((1,) * len(fns_shape) + (3,), (*fns_shape, 3)),
        new_axis=[len(fns_shape)],
        meta=np.array([], dtype=float),
    )

    positions_xr = xr.DataArray(
        positions_da,
        name="stage_positions",
        dims=(*fns_xr.dims, "pos"),
        coords=fns_xr.coords,
    )
    positions_xr.coords.update({"pos": ["pos_z", "pos_y", "pos_x"]})

    return positions_xr


def _read_images(x: ArrayLike, ny: int, nx: int, im_dtype: type) -> ArrayLike:
    """Function to read images from filenames."""
    images = np.zeros((*x.shape, ny, nx), dtype=im_dtype)
    for i in np.ndindex(x.shape):
        filename = x[i]
        if filename != "":
            images[i] = tifffile.imread(filename)
    return images


def lazy_load_images(fns_xr: xr.DataArray) -> xr.DataArray:
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
    dtype = image.dtype

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
        attrs=_build_channel_metadata(fns_xr),
    )

    return images_xr


# TODO: write tests for this function
def lazy_load_plate(fns_xr: xr.DataArray) -> xr.Dataset:
    """
    Lazily load the plate data (images & stage postitions & image coordinates).

    Args:
        fns_xr (xr.DataArray): An xarray containing the filenames in the
            correct structure.

    Returns:
        xr.Dataset: An xarray dataset containing the images and stage
            positions. The condensed stage coordinates and time points are
            included as coordinates.
    """
    images_xr = lazy_load_images(fns_xr)
    positions_xr = lazy_load_stage_positions(fns_xr)

    # compute the x and y coordinates for the stage positions only once per
    # time, channel, and plane (assume that all are roughly the same)
    coords_x = (
        positions_xr.isel(time=0, channel=0, plane=0, drop=True)
        .sel(pos="pos_x")
        .compute()
    )
    coords_y = (
        positions_xr.isel(time=0, channel=0, plane=0, drop=True)
        .sel(pos="pos_y")
        .compute()
    )
    # compute the z and t coordinates from the estimated dz and dt
    dz = images_xr.attrs[images_xr.coords["channel"].values[0]]["dz"]
    coords_z = xr.DataArray(
        data=np.array([n * dz for n in range(len(images_xr.coords["plane"]))]),
        coords={"plane": positions_xr.coords["plane"]},
    )
    dt = images_xr.attrs[images_xr.coords["channel"].values[0]]["dt"]
    coords_t = xr.DataArray(
        data=np.array([n * dt for n in range(len(images_xr.coords["time"]))]),
        coords={"time": positions_xr.coords["time"]},
    )

    data = xr.Dataset(
        {images_xr.name: images_xr, positions_xr.name: positions_xr},
        coords={
            "coords_x": coords_x,
            "coords_y": coords_y,
            "coords_z": coords_z,
            "coords_t": coords_t,
        },
    )

    return data
