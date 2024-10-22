import time
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Union

import dask
import tifffile
import xarray as xr
from dask.diagnostics import ProgressBar

from zmb_md_converter.io.assembly import (
    create_filename_structure_MD,
    lazy_load_plate,
)
from zmb_md_converter.io.parsing import (
    _fill_mixed_acquisitions,
    parse_MD_plate_folder,
    parse_MD_tz_folder,
)


def _well_output_name(dataset: xr.Dataset) -> str:
    if dataset.well.size != 1:
        raise ValueError("Well dimension must have size 1")
    return str(dataset.well.values.item())


def _field_output_name(dataset: xr.Dataset, legacy_mode: bool = False) -> str:
    if dataset.field.size != 1:
        raise ValueError("Field dimension must have size 1")
    coord_str = dataset.field.values[0]
    if legacy_mode:
        name = f"_f_{int(coord_str[1:]):03d}"
    else:
        name = f"f{int(coord_str[1:]):03d}"
    return name


def _time_output_name(dataset: xr.Dataset) -> str:
    if dataset.time.size != 1:
        raise ValueError("Time dimension must have size 1")
    coord_str = dataset.time.values[0]
    name = f"t{int(coord_str):03d}"
    return name


def _channel_output_name(dataset: xr.Dataset) -> str:
    if dataset.channel.size != 1:
        raise ValueError("Channel dimension must have size 1")
    coord_str = dataset.channel.values[0]
    name = f"c{int(coord_str[1:]):02d}"
    return name


def _plane_output_name(dataset: xr.Dataset) -> str:
    if dataset.plane.size != 1:
        raise ValueError("Plane dimension must have size 1")
    coord_str = dataset.plane.values[0]
    name = f"z{int(coord_str):03d}"
    return name


def _plate_output_name(dataset: xr.Dataset) -> str:
    example_channel_metadata = next(iter(dataset.images.attrs.items()))[1]
    return str(example_channel_metadata["plate_name"])


def _get_dimension_sizes(dataset: xr.Dataset) -> list:
    """
    Get dimension sizes of dataset.

    Will return size for "well", "field", "time", "channel", "plane", "y", "x".
    If a dimension is not present in the dataset, the size will be 1.
    """
    all_dims_index = dict(
        zip(["well", "field", "time", "channel", "plane", "y", "x"], range(7))
    )
    dim_sizes = list(range(7))
    for dim, size in zip(dataset.images.dims, dataset.images.shape):
        dim_sizes[all_dims_index[dim]] = size
    return dim_sizes


def _get_tczyx_ome_metadata(dataset: xr.Dataset) -> dict:
    """Produce metadata for TCZYX ome-tiff export with tifffile."""
    nw, nf, nt, nc, nz, ny, nx = _get_dimension_sizes(dataset)
    example_channel_metadata = next(iter(dataset.images.attrs.items()))[1]

    # tifffile seems to require dz to be > 0
    if example_channel_metadata["dz"] == 0:
        example_channel_metadata["dz"] = 1

    output_metadata = {
        "axes": "TCZYX",
        "TimeIncrement": example_channel_metadata["dt"],
        "TimeIncrementUnit": example_channel_metadata["time_calibration_units"],
        "PhysicalSizeX": example_channel_metadata["dx"],
        "PhysicalSizeXUnit": example_channel_metadata["spatial_calibration_units"],
        "PhysicalSizeY": example_channel_metadata["dy"],
        "PhysicalSizeYUnit": example_channel_metadata["spatial_calibration_units"],
        "PhysicalSizeZ": example_channel_metadata["dz"],
        "PhysicalSizeZUnit": example_channel_metadata["spatial_calibration_units"],
        "Channel": {
            "Name": [
                dataset.images.attrs[key]["channel_name"]
                for key in dataset.images.coords["channel"].data
            ]
        },
        "Plane": {
            "PositionX": [dataset.coords_x.item() for n in range(nt * nc * nz)],
            "PositionY": [dataset.coords_y.item() for n in range(nt * nc * nz)],
            "PositionZ": list(dataset.coords_z.data) * nt * nc,
        },
    }

    return output_metadata


def _write_tczyx(dataset: xr.Dataset, tiffwriter: tifffile.TiffWriter) -> None:
    """
    Write a dataset with dimensions TCZYX to an open tifffile.TiffWriter.

    Args:
        dataset (xr.Dataset): An xarray dataset containing the image and stage
            position to be saved. The image must have (only) the dimensions
            'time', 'channel', 'plane', 'y', 'x'.
        tiffwriter (tifffile.TiffWriter): An open tifffile.TiffWriter object.
    """
    expected_dims = ("time", "channel", "plane", "y", "x")
    if set(expected_dims) != set(dataset.images.dims):
        raise ValueError(
            f"Dataset must (only) contain {expected_dims}, "
            f"but actually contains {dataset.images.dims}."
        )

    tiffwriter.write(
        dataset.images.data,
        photometric="minisblack",
        metadata=_get_tczyx_ome_metadata(dataset),
    )


def save_well_as_ome_tiffs(
    dataset: xr.Dataset,
    out_dir: Union[str, Path],
    split: Iterable[str] = ("field",),
) -> None:
    """
    Save the dataset from one well as (multiple) .ome.tif stacks.

    Args:
        dataset (xr.Dataset): An xarray dataset containing the images and
            stage positions to be saved. The images must have (only) the
            dimensions 'field', 'time', 'channel', 'plane', 'y', 'x'.
        out_dir (Union[str, Path]): The directory where the .ome.tif stacks
            will be saved.
        split (Iterable[str]): An iterable containing the dimensions along
            which the data will be split. E.g. if split=('field',), an
            individual stack will be saved for each field. Multiple dimensions
            can be split. They must be a subset of ('field', 'time', 'channel',
            'plane').
    """
    expected_dims = ("field", "time", "channel", "plane", "y", "x")
    if set(expected_dims) != set(dataset.images.dims):
        raise ValueError(
            f"Dataset must (only) contain {expected_dims}, "
            f"but actually contains {dataset.images.dims}."
        )
    if not all(s in ("field", "time", "channel", "plane") for s in split):
        raise ValueError(
            "Split dimensions must be a subset of "
            "('field', 'time', 'channel', 'plane'), "
            f"but actually contains {split}."
        )

    split_coords = [dataset.images[dim].values for dim in split]

    # setup delayed execution:
    @dask.delayed  # type: ignore
    def _save_tczyx_delayed(
        dataset_sub: xr.Dataset, out_dir: Union[str, Path], fn_output: str
    ) -> None:
        with tifffile.TiffWriter(
            Path(out_dir) / fn_output, bigtiff=(dataset_sub.images.nbytes >= 2**32)
        ) as tiffwriter:
            _write_tczyx(dataset_sub, tiffwriter)

    @dask.delayed  # type: ignore
    def _save_ftczyx_delayed(
        dataset_sub: xr.Dataset, out_dir: Union[str, Path], fn_output: str
    ) -> None:
        with tifffile.TiffWriter(
            Path(out_dir) / fn_output, bigtiff=(dataset_sub.images.nbytes >= 2**32)
        ) as tiffwriter:
            for field in dataset_sub.field.values:
                dataset_sub_sub = dataset_sub.sel(field=field)
                _write_tczyx(dataset_sub_sub, tiffwriter)

    delayed_objects = []

    # loop over all combinations of the coordinates to split
    for combination in product(*split_coords):
        dataset_sub = dataset.sel(
            **{dim: [coord] for dim, coord in zip(split, combination)}
        )
        fn_output = (
            f"{_plate_output_name(dataset_sub)}"
            f"_{_well_output_name(dataset_sub)}"
            f"{'_' + _field_output_name(dataset_sub) if 'field' in split else ''}"
            f"{'_' + _time_output_name(dataset_sub) if 'time' in split else ''}"
            f"{'_' + _channel_output_name(dataset_sub) if 'channel' in split else ''}"
            f"{'_' + _plane_output_name(dataset_sub) if 'plane' in split else ''}"
            ".ome.tiff"
        )
        if "field" in split:
            # drop 'field' dimension (without loosing the coordinate)
            dataset_sub = dataset_sub.sel(field=dataset_sub.field.values.item())
            delayed_objects.append(_save_tczyx_delayed(dataset_sub, out_dir, fn_output))
        else:
            # special case field: write fields in one file, but as separate series
            delayed_objects.append(
                _save_ftczyx_delayed(dataset_sub, out_dir, fn_output)
            )

    # execute delayed objects
    with ProgressBar():
        dask.compute(*delayed_objects)


def save_well_as_imagej_tiffs_legacy(
    dataset: xr.Dataset,
    out_dir: Union[str, Path],
) -> None:
    """
    Save the dataset from one well as imageJ hyperstacks (splitting fields).

    This is the legacy mode, roughly reproducing the 'MDParallel2Hyperstck_v3'
    script.

    Args:
        dataset (xr.Dataset): An xarray dataset containing the images and
            stage positions to be saved. The images must have (only) the
            dimensions 'field', 'time', 'channel', 'plane', 'y', 'x'.
        out_dir (Union[str, Path]): The directory where the .ome.tif stacks
            will be saved.
    """
    expected_dims = ("field", "time", "channel", "plane", "y", "x")
    if set(expected_dims) != set(dataset.images.dims):
        raise ValueError(
            f"Dataset must (only) contain {expected_dims}, "
            f"but actually contains {dataset.images.dims}."
        )

    nfields = len(dataset.images["field"].values)

    # setup delayed execution:
    @dask.delayed  # type: ignore
    def _save_tzcyx_delayed(
        dataset_sub: xr.Dataset,
        out_dir: Union[str, Path],
        fn_output: str,
        ome_metadata: dict,
    ) -> None:
        tifffile.imwrite(
            Path(out_dir) / fn_output,
            dataset_sub.images.data.swapaxes(1, 2),
            imagej=True,
            photometric="minisblack",
            resolution=(
                1 / ome_metadata["PhysicalSizeX"],
                1 / ome_metadata["PhysicalSizeY"],
            ),
            metadata={
                "axes": "TZCYX",
                "spacing": ome_metadata["PhysicalSizeZ"],
                "unit": "um",
                "finterval": ome_metadata["TimeIncrement"],
            },
        )

    delayed_objects = []

    # loop over all fields
    for field in dataset.images["field"].values:
        dataset_sub = dataset.sel(field=[field])
        fn_output = (
            f"{_plate_output_name(dataset_sub)}"
            f"_w_{_well_output_name(dataset_sub)}"
            f"{_field_output_name(dataset_sub, legacy_mode=True) if nfields>1 else ''}"
            ".tif"
        )
        # drop 'field' dimension (without loosing the coordinate)
        dataset_sub = dataset_sub.sel(field=dataset_sub.field.values.item())
        ome_metadata = _get_tczyx_ome_metadata(dataset_sub)
        delayed_objects.append(
            _save_tzcyx_delayed(dataset_sub, out_dir, fn_output, ome_metadata)
        )

    # execute delayed objects
    with ProgressBar():
        dask.compute(*delayed_objects)


def convert_md_to_ome_tiffs(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    only_2D: bool = False,
    dimensions_to_split: Iterable[str] = ("field",),
    fill_mixed_acquisition: bool = True,
) -> None:
    """
    Convert MD-data to .ome.tif stacks.

    Args:
        input_dir (Union[str, Path]): Path to the top folder of the MD-data.
            In case of a single plate acquisition: Should contain the
            'Timepoint_*' folders. In case of a combined time & z-stack
            measurement: Should contain the different MD-plate folders.
        output_dir (Union[str, Path]): Path to the output directory where the
            .ome.tif stacks will be saved.
        only_2D (bool): If True, only the 2D images (projections) will be
            saved. If False, the 3D stacks will be saved (discarding any
            projections).
        dimensions_to_split (Iterable[str]): An iterable containing the
            dimensions along which the data will be split. E.g. if
            split=('field'), an individual stack will be saved for each field.
            Multiple dimensions can be split. They must be a subset of
            ('field', 'time', 'channel', 'plane').
        fill_mixed_acquisition (bool): If True, missing planes and timepoints
            will be filled with existing data. (E.g. if only one plane was
            acquired in a channel, the entire stack will be filled with that
            plane. Or if only every nth timepoint was acquired for a channel,
            the missing timepoints will be filled with the last acquired.) This
            is already done if the data was exported from the MetaXpress
            software, but not if the data was moved directly from the
            microscope.
    """
    print(f"input path: {input_dir}")
    print(f"output path: {output_dir}")
    print("\nReading plate data structure...")

    t_start = time.time()

    files_df = parse_MD_plate_folder(input_dir)
    if files_df is None:
        files_df = parse_MD_tz_folder(input_dir)
    if files_df is None:
        raise ValueError("No files found in the input directory.")

    if fill_mixed_acquisition:
        files_df = _fill_mixed_acquisitions(files_df)

    fns_xr = create_filename_structure_MD(files_df, only_2D=only_2D)
    dataset = lazy_load_plate(fns_xr)

    print("\nConverting plate:")
    example_channel_metadata = next(iter(dataset.images.attrs.items()))[1]
    print(
        f"plate_name: {example_channel_metadata['plate_name']}\n"
        f"dx: {example_channel_metadata['dx']}"
        f" {example_channel_metadata['spatial_calibration_units']}\n"
        f"dy: {example_channel_metadata['dy']}"
        f" {example_channel_metadata['spatial_calibration_units']}\n"
        f"dz: {example_channel_metadata['dz']}"
        f" {example_channel_metadata['spatial_calibration_units']}\n"
        f"dt: {example_channel_metadata['dt']}"
        f" {example_channel_metadata['time_calibration_units']}"
    )
    print("Dimensions:")
    print(dict(zip(dataset.images.dims, dataset.images.shape)))

    nwells = len(dataset.well.values)
    for w, well in enumerate(dataset.well.values):
        print(f"\nProcessing well {well} ({w+1}/{nwells})")
        well_dataset = dataset.sel(well=well)
        save_well_as_ome_tiffs(well_dataset, output_dir, split=dimensions_to_split)

    t_end = time.time()
    print(f"\nDone. Conversion took {t_end - t_start:.1f} seconds.\n")


def convert_md_to_imagej_hyperstacks(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """
    Convert MD-data to imagej hyperstacks.

    This is the legacy mode, which mimics the 'MDParallel2Hyperstck_v3' script.
    Key differences to the 'MDParallel2Hyperstck_v3' script:
    - Projections will not be saved in the same stack as the rest of the 3D
    data.
    - z-spacing will be estimated from the stage-positions, instead of set to
    5*dx.

    Compared to the 'convert_md_to_ome_tiffs' function, this function will set
    only_2D=False
    dimensions_to_split=('field',)
    fill_mixed_acquisition=False

    Args:
        input_dir (Union[str, Path]): Path to the top folder of the MD-data.
            In case of a single plate acquisition: Should contain the
            'Timepoint_*' folders. In case of a combined time & z-stack
            measurement: Should contain the different MD-plate folders.
        output_dir (Union[str, Path]): Path to the output directory where the
            .tif stacks will be saved.
    """
    print(f"input path: {input_dir}")
    print(f"output path: {output_dir}")
    print("\nReading plate data structure...")

    t_start = time.time()

    files_df = parse_MD_plate_folder(input_dir)
    if files_df is None:
        files_df = parse_MD_tz_folder(input_dir)
    if files_df is None:
        raise ValueError("No files found in the input directory.")

    files_df = _fill_mixed_acquisitions(files_df)

    fns_xr = create_filename_structure_MD(files_df, only_2D=False)
    dataset = lazy_load_plate(fns_xr)

    print("\nConverting plate:")
    example_channel_metadata = next(iter(dataset.images.attrs.items()))[1]
    print(
        f"plate_name: {example_channel_metadata['plate_name']}\n"
        f"dx: {example_channel_metadata['dx']}"
        f" {example_channel_metadata['spatial_calibration_units']}\n"
        f"dy: {example_channel_metadata['dy']}"
        f" {example_channel_metadata['spatial_calibration_units']}\n"
        f"dz: {example_channel_metadata['dz']}"
        f" {example_channel_metadata['spatial_calibration_units']}\n"
        f"dt: {example_channel_metadata['dt']}"
        f" {example_channel_metadata['time_calibration_units']}"
    )
    print("Dimensions:")
    print(dict(zip(dataset.images.dims, dataset.images.shape)))

    nwells = len(dataset.well.values)
    for w, well in enumerate(dataset.well.values):
        print(f"\nProcessing well {well} ({w+1}/{nwells})")
        well_dataset = dataset.sel(well=well)
        save_well_as_imagej_tiffs_legacy(well_dataset, output_dir)

    t_end = time.time()
    print(f"\nDone. Conversion took {t_end - t_start:.1f} seconds.\n")
