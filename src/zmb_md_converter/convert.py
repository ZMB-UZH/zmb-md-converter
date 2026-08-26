import re
import time
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Literal, Union

import dask
import numpy as np
import tifffile
import xarray as xr
from dask.diagnostics import ProgressBar

from zmb_md_converter.io.hcs_ai.assembly import (
    create_well_filename_structure_MD as create_well_filename_structure_MD_hcs_ai,
)
from zmb_md_converter.io.hcs_ai.assembly import (
    lazy_load_well as lazy_load_well_hcs_ai,
)
from zmb_md_converter.io.hcs_ai.parsing import (
    parse_MD_plate_folder as parse_MD_plate_folder_hcs_ai,
)
from zmb_md_converter.io.ht_ai.assembly import (
    create_well_filename_structure_MD,
    lazy_load_well,
)
from zmb_md_converter.io.ht_ai.parsing import (
    _fill_mixed_acquisitions,
    parse_MD_plate_folder,
    parse_MD_tz_folder,
)

_INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize_output_name(name: str) -> str:
    r"""
    Sanitize a string for use as part of an output filename.

    Any run of whitespace, punctuation, or other characters that are not
    safe to use in filenames (e.g. on Windows: ``<>:"/\\|?*``) is replaced
    by a single underscore, and leading/trailing underscores are stripped.
    """
    return _INVALID_FILENAME_CHARS.sub("_", name).strip("_")


def _well_output_name(dataset: xr.Dataset) -> str:
    example_channel_metadata = next(iter(dataset.images.attrs.items()))[1]
    return _sanitize_output_name(str(example_channel_metadata["well_name"]))


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
    return _sanitize_output_name(str(example_channel_metadata["plate_name"]))


def _get_tczyx_ome_metadata(dataset: xr.Dataset) -> dict:
    """Produce metadata for TCZYX ome-tiff export with tifffile."""
    nt = dataset.images.sizes["time"]
    nc = dataset.images.sizes["channel"]
    nz = dataset.images.sizes["plane"]
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
            f"{_field_output_name(dataset_sub, legacy_mode=True) if nfields > 1 else ''}"
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


def z_project_dataset(
    dataset: xr.Dataset, method: Literal["max", "mean"] = "max"
) -> xr.Dataset:
    """
    Z-project the dataset using the specified method.

    Args:
        dataset (xr.Dataset): The dataset to be z-projected.
        method (str): The method to use for z-projection. Can be either
            "max" or "mean".

    Returns:
        xr.Dataset: The z-projected dataset.
    """
    # project images
    if method == "max":
        images_xr = dataset.images.max(dim="plane", keepdims=True, keep_attrs=True)
    elif method == "mean":
        images_xr = dataset.images.mean(dim="plane", keepdims=True, keep_attrs=True)
    else:
        raise ValueError("Method must be either 'max' or 'mean'.")

    # take mean of stage positions along z
    positions_xr = dataset.stage_positions.mean(
        dim="plane", keepdims=True, keep_attrs=True
    )

    # create new coordinate for plane
    images_xr = images_xr.assign_coords(plane=["0"])
    positions_xr = positions_xr.assign_coords(plane=["0"])

    # create new stage position coordinate
    coords_z = xr.DataArray(
        data=np.array([0]),
        coords={"plane": positions_xr.coords["plane"]},
    )

    dataset = xr.Dataset(
        {images_xr.name: images_xr, positions_xr.name: positions_xr},
        coords={
            "coords_z": coords_z,
        },
    )

    return dataset


def convert_ht_ai_to_tiffs(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    only_2D: bool = False,
    dimensions_to_split: Iterable[str] = ("field",),
    fill_mixed_acquisition: bool = True,
    legacy_mode: bool = False,
) -> None:
    """
    Convert MD-data to TIF stacks.

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
        legacy_mode (bool): If True, the function will save the data in a
            legacy format, which mimics the 'MDParallel2Hyperstck_v3' script.
            Key differences to the 'MDParallel2Hyperstck_v3' script:
                - Projections will not be saved in the same stack as the rest
                of the 3D data.
                - z-spacing will be estimated from the stage-positions, instead
                of set to 5*dx.
            This mode only works with the settings:
                only_2D=False
                dimensions_to_split=('field',)
                fill_mixed_acquisition=False

    """
    if legacy_mode:
        if only_2D:
            only_2D = False
            print(
                "Legacy mode is not compatible with only_2D=True. "
                "Setting only_2D=False."
            )
        if dimensions_to_split != ("field",):
            dimensions_to_split = ("field",)
            print(
                "Legacy mode is not compatible with dimensions_to_split != ('field',). "
                "Setting dimensions_to_split=('field',)."
            )
        if fill_mixed_acquisition:
            fill_mixed_acquisition = False
            print(
                "Legacy mode is not compatible with fill_mixed_acquisition=True. "
                "Setting fill_mixed_acquisition=False."
            )

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

    print(f"\nConverting plate '{files_df['name'].unique()[0]}'...")

    nwells = len(files_df["well"].unique())
    failed_wells: list[tuple[str, Exception]] = []
    for w, well in enumerate(sorted(files_df["well"].unique())):
        print(f"\nProcessing well {well} ({w + 1}/{nwells})")
        well_files_df = files_df[files_df["well"] == well]
        try:
            try:
                fns_xr = create_well_filename_structure_MD(
                    well_files_df, only_2D=only_2D
                )
                dataset = lazy_load_well(fns_xr)
            except ValueError as e:
                # if the well has no projection data, create it from the 3D data
                if "No projections found." in str(e):
                    print(
                        "No projections found. "
                        "Creating max intensity projections from 3D data..."
                    )
                    well_files_df = _fill_mixed_acquisitions(well_files_df)
                    fns_xr = create_well_filename_structure_MD(
                        well_files_df, only_2D=False
                    )
                    dataset = lazy_load_well(fns_xr)
                    dataset = z_project_dataset(dataset, method="max")
                else:
                    raise

            example_channel_metadata = next(iter(dataset.images.attrs.items()))[1]
            print(
                f"dx: {example_channel_metadata['dx']}"
                f" {example_channel_metadata['spatial_calibration_units']}\t"
                f"dy: {example_channel_metadata['dy']}"
                f" {example_channel_metadata['spatial_calibration_units']}\t"
                f"dz: {example_channel_metadata['dz']}"
                f" {example_channel_metadata['spatial_calibration_units']}\t"
                f"dt: {example_channel_metadata['dt']}"
                f" {example_channel_metadata['time_calibration_units']}"
            )
            print(f"Dimensions: {dict(dataset.images.sizes)}")

            if legacy_mode:
                save_well_as_imagej_tiffs_legacy(dataset, output_dir)
            else:
                save_well_as_ome_tiffs(dataset, output_dir, split=dimensions_to_split)
        except Exception as e:
            print(f"  ERROR processing well {well}: {e}")
            failed_wells.append((well, e))

    t_end = time.time()
    print(f"\nDone. Conversion took {t_end - t_start:.1f} seconds.")
    if failed_wells:
        summary = ", ".join(f"{w}: {e}" for w, e in failed_wells)
        raise RuntimeError(
            f"{len(failed_wells)}/{nwells} well(s) failed conversion: {summary}"
        )
    print()


def convert_hcs_ai_to_tiffs(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    only_2D: bool = False,
    use_montage: bool = True,
    dimensions_to_split: Iterable[str] = ("field",),
) -> None:
    """
    Convert MD ImageXpress HCS.ai data to OME-TIFF stacks.

    Args:
        input_dir: Path to the HCS.ai acquisition folder (the one containing
            the .mxprotocol file).
        output_dir: Path to the output directory where the .ome.tif stacks
            will be saved.
        only_2D: If True, save 2D projections rather than 3D stacks.  When
            no pre-calculated projections are available, max-intensity
            projections are computed automatically from the 3D data.
        use_montage: If True (default), prefer stitched images from
            ``experiment_montage/`` over the per-tile data.  When the montage
            is a z-stack and ``only_2D=True``, projections are calculated
            from the stitched stack.
        dimensions_to_split: Dimensions along which to split into separate
            output files.  Must be a subset of
            ('field', 'time', 'channel', 'plane').

    Fallback behaviour
    ------------------
    - ``only_2D=False, use_montage=False``, no z-stack present
      → returns 2D projection data automatically.
    - ``only_2D=True, use_montage=False``, only z-stack present
      → loads z-stack and computes max-intensity projections.
    - ``only_2D=False, use_montage=True``, montage is a single projection
      → returns the 2D montage image.
    - ``only_2D=True, use_montage=True``, montage is a z-stack
      → loads the stitched stack and computes max-intensity projections.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"input path: {input_dir}")
    print(f"output path: {output_dir}")
    print("\nReading plate data structure...")

    t_start = time.time()

    # --- Parse with fallback for missing-folder cases ---
    try:
        files_df = parse_MD_plate_folder_hcs_ai(
            input_dir, only_2D=only_2D, use_montage=use_montage
        )
    except FileNotFoundError:
        if only_2D and not use_montage:
            # Case: only_2D=True but no experiment/ folder (only z-stack exists).
            # Load the z-stack; projections will be calculated per-well below.
            print(
                "No projection folder found. "
                "Will load z-stack data and calculate max-intensity projections..."
            )
            files_df = parse_MD_plate_folder_hcs_ai(
                input_dir, only_2D=False, use_montage=False
            )
        else:
            raise

    if files_df is None:
        raise ValueError(
            f"No HCS.ai acquisition found in {input_dir}. "
            "(No .mxprotocol file detected.)"
        )

    print(f"\nConverting plate '{files_df['name'].unique()[0]}'...")

    nwells = len(files_df["well"].unique())
    failed_wells: list[tuple[str, Exception]] = []
    for w, well in enumerate(sorted(files_df["well"].unique())):
        print(f"\nProcessing well {well} ({w + 1}/{nwells})")
        well_files_df = files_df[files_df["well"] == well]
        try:
            needs_projection = False
            try:
                fns_xr = create_well_filename_structure_MD_hcs_ai(
                    well_files_df, only_2D=only_2D
                )
            except ValueError as e:
                if "No projections found." in str(e) and only_2D:
                    # Cases: data is 3D (z-stack or montage z-stack) but projections
                    # were requested → load the stack and compute them.
                    print(
                        "No pre-calculated projections found. "
                        "Calculating max-intensity projections from 3D data..."
                    )
                    fns_xr = create_well_filename_structure_MD_hcs_ai(
                        well_files_df, only_2D=False
                    )
                    needs_projection = True
                else:
                    raise

            dataset = lazy_load_well_hcs_ai(fns_xr)

            if needs_projection:
                dataset = z_project_dataset(dataset, method="max")

            example_channel_metadata = next(iter(dataset.images.attrs.items()))[1]
            print(
                f"dx: {example_channel_metadata['dx']}"
                f" {example_channel_metadata['spatial_calibration_units']}\t"
                f"dy: {example_channel_metadata['dy']}"
                f" {example_channel_metadata['spatial_calibration_units']}\t"
                f"dz: {example_channel_metadata['dz']}"
                f" {example_channel_metadata['spatial_calibration_units']}\t"
                f"dt: {example_channel_metadata['dt']}"
                f" {example_channel_metadata['time_calibration_units']}"
            )
            print(f"Dimensions: {dict(dataset.images.sizes)}")

            save_well_as_ome_tiffs(dataset, output_dir, split=dimensions_to_split)
        except Exception as e:
            print(f"  ERROR processing well {well}: {e}")
            failed_wells.append((well, e))

    t_end = time.time()
    print(f"\nDone. Conversion took {t_end - t_start:.1f} seconds.")
    if failed_wells:
        summary = ", ".join(f"{w}: {e}" for w, e in failed_wells)
        raise RuntimeError(
            f"{len(failed_wells)}/{nwells} well(s) failed conversion: {summary}"
        )
    print()
