"""Tests for MD ImageXpress HCS.ai assembly functions."""

import numpy as np
import pytest
import xarray as xr

from zmb_md_converter.io.hcs_ai.assembly import (
    create_well_filename_structure_MD,
    lazy_load_well,
)
from zmb_md_converter.io.hcs_ai.parsing import parse_MD_plate_folder


@pytest.fixture(scope="module")
def well_files(hcs_ai_temp_dir):
    data_dir = (
        hcs_ai_temp_dir
        / "1w4s1t1c4z_binning4x4_tiled-L_montage-stitch-zstack"
        / "test_data_20260316_145602"
    )
    df = parse_MD_plate_folder(data_dir)
    return df[df["well"] == "C05"]


@pytest.fixture(scope="module")
def fns_xr(well_files):
    return create_well_filename_structure_MD(well_files)


@pytest.fixture(scope="module")
def fns_xr_2d(well_files):
    return create_well_filename_structure_MD(well_files, only_2D=True)


def test_fns_xr_dims(fns_xr):
    assert set(fns_xr.dims) == {"field", "time", "channel", "plane"}


def test_fns_xr_coords_3d(fns_xr):
    # 4 fields (s0-s3 from z-stack), 1 channel (w0), 4 z-planes
    assert fns_xr.sizes["plane"] == 4
    assert fns_xr.sizes["channel"] == 1


def test_fns_xr_only_2d(fns_xr_2d):
    assert fns_xr_2d.sizes["plane"] == 1
    assert fns_xr_2d.coords["plane"].values[0] == "0"


def test_fns_xr_attrs_contain_metadata(fns_xr):
    assert "channel_metadata" in fns_xr.attrs
    assert "stage_positions" in fns_xr.attrs
    assert "plate_name" in fns_xr.attrs
    assert "well_name" in fns_xr.attrs


def test_channel_metadata_keys(fns_xr):
    ch_meta = fns_xr.attrs["channel_metadata"]
    expected = {
        "plate_name",
        "well_name",
        "channel_name",
        "dx",
        "dy",
        "dz",
        "dt",
        "spatial_calibration_units",
        "time_calibration_units",
    }
    for meta in ch_meta.values():
        assert expected.issubset(set(meta.keys()))


def test_stage_positions_shape(fns_xr):
    pos = fns_xr.attrs["stage_positions"]
    expected_shape = (*fns_xr.shape, 3)
    assert pos.shape == expected_shape


def test_lazy_load_well_structure(fns_xr):
    dataset = lazy_load_well(fns_xr)

    assert isinstance(dataset, xr.Dataset)
    assert "images" in dataset
    assert "stage_positions" in dataset
    assert "coords_x" in dataset.coords
    assert "coords_y" in dataset.coords
    assert "coords_z" in dataset.coords
    assert "coords_t" in dataset.coords


def test_lazy_load_well_image_dims(fns_xr):
    dataset = lazy_load_well(fns_xr)

    assert set(dataset.images.dims) == {"field", "time", "channel", "plane", "y", "x"}


def test_lazy_load_well_image_loads(fns_xr):
    dataset = lazy_load_well(fns_xr)

    # Load a single image to verify dask graph executes correctly
    img = dataset.images.isel(field=0, time=0, channel=0, plane=0).compute()
    assert img.shape == (576, 576)
    assert img.dtype == np.uint16
