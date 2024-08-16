# TODO: Expand tests to cover more MetaXpress exports and include timeseries folder.

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from zmb_md_converter.io.assembly import (
    _build_channel_metadata,
    _check_if_channel_contains_stack,
    _check_if_channel_contains_timeseries,
    _get_t_spacing,
    _get_z_spacing,
    _read_images,
    _read_stage_positions,
    create_filename_structure_MD,
    lazy_load_images,
    lazy_load_stage_positions,
)
from zmb_md_converter.parsing import parse_MD_plate_folder


def test_create_filename_structure_MD(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert fns_xr.shape == (1, 1, 1, 1, 1)
    assert fns_xr.well.values.tolist() == ["B02"]
    assert fns_xr.field.values.tolist() == ["s1"]
    assert fns_xr.time.values.tolist() == ["1"]
    assert fns_xr.channel.values.tolist() == ["w1"]
    assert fns_xr.plane.values.tolist() == ["0"]
    assert fns_xr.attrs == {"plate_name": "9987"}
    assert "" not in fns_xr

    # 1t-3z-2w-2s-2c
    root_dir = temp_dir / "direct_transfer" / "3433"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert fns_xr.shape == (2, 2, 1, 2, 3)
    assert fns_xr.well.values.tolist() == ["B02", "B03"]
    assert fns_xr.field.values.tolist() == ["s1", "s2"]
    assert fns_xr.time.values.tolist() == ["1"]
    assert fns_xr.channel.values.tolist() == ["w1", "w2"]
    assert fns_xr.plane.values.tolist() == ["1", "2", "3"]
    assert fns_xr.attrs == {"plate_name": "9987"}
    assert "" not in fns_xr

    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert fns_xr.shape == (2, 2, 1, 3, 3)
    assert fns_xr.well.values.tolist() == ["B02", "B03"]
    assert fns_xr.field.values.tolist() == ["s1", "s2"]
    assert fns_xr.time.values.tolist() == ["1"]
    assert fns_xr.channel.values.tolist() == ["w1", "w3", "w4"]
    assert fns_xr.plane.values.tolist() == ["1", "2", "3"]
    assert fns_xr.attrs == {"plate_name": "9987"}
    assert "" not in fns_xr[0, 0, 0, 0]
    assert fns_xr[0, 0, 0, 1, 0] != ""
    assert all(fns_xr[0, 0, 0, 1, 1:] == ["", ""])

    # 1t-3z-2w-2s-4c mixed z-sampliing - 3D
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df, only_2D=True)
    assert fns_xr.shape == (2, 2, 1, 2, 1)
    assert fns_xr.well.values.tolist() == ["B02", "B03"]
    assert fns_xr.field.values.tolist() == ["s1", "s2"]
    assert fns_xr.time.values.tolist() == ["1"]
    assert fns_xr.channel.values.tolist() == ["w1", "w2"]
    assert fns_xr.plane.values.tolist() == ["0"]
    assert fns_xr.attrs == {"plate_name": "9987"}
    assert "" not in fns_xr

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "direct_transfer" / "3435"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert fns_xr.shape == (2, 2, 6, 4, 1)
    assert fns_xr.well.values.tolist() == ["B02", "B03"]
    assert fns_xr.field.values.tolist() == ["s1", "s2"]
    assert fns_xr.time.values.tolist() == ["1", "2", "3", "4", "5", "6"]
    assert fns_xr.channel.values.tolist() == ["w1", "w2", "w3", "w4"]
    assert fns_xr.plane.values.tolist() == ["0"]
    assert fns_xr.attrs == {"plate_name": "9987"}
    assert "" not in fns_xr[0, 0, :, 0, 0]
    assert fns_xr[0, 0, 0, 1, 0] != ""
    assert all(fns_xr[0, 0, 1:, 1, 0] == ["", "", "", "", ""])

    # check for 'no projections found' error:
    with pytest.raises(ValueError):
        create_filename_structure_MD(files_df[files_df.z.notnull()], only_2D=True)

    # check for 'multiple files found' error:
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = parse_MD_plate_folder(root_dir)
    with pytest.raises(RuntimeError):
        create_filename_structure_MD(pd.concat([files_df, files_df]))

    # MD export
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert fns_xr.shape == (1, 1, 1, 1, 1)
    assert fns_xr.well.values.tolist() == ["B02"]
    assert fns_xr.field.values.tolist() == ["s1"]
    assert fns_xr.time.values.tolist() == ["1"]
    assert fns_xr.channel.values.tolist() == ["w1"]
    assert fns_xr.plane.values.tolist() == ["0"]
    assert fns_xr.attrs == {"plate_name": "9987"}
    assert "" not in fns_xr


def test_check_if_channel_contains_stack(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert not _check_if_channel_contains_stack(fns_xr, 0)

    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert _check_if_channel_contains_stack(fns_xr, 0)
    assert not _check_if_channel_contains_stack(fns_xr, 1)
    assert not _check_if_channel_contains_stack(fns_xr, 2)

    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3434"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert _check_if_channel_contains_stack(fns_xr, 0)
    assert not _check_if_channel_contains_stack(fns_xr, 1)
    assert not _check_if_channel_contains_stack(fns_xr, 2)
    assert not _check_if_channel_contains_stack(fns_xr, 3)


def test_check_if_channel_contains_timeseries(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert not _check_if_channel_contains_timeseries(fns_xr, 0)

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "direct_transfer" / "3435"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert _check_if_channel_contains_timeseries(fns_xr, 0)
    assert not _check_if_channel_contains_timeseries(fns_xr, 1)
    assert not _check_if_channel_contains_timeseries(fns_xr, 2)
    assert not _check_if_channel_contains_timeseries(fns_xr, 3)

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3435"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert _check_if_channel_contains_timeseries(fns_xr, 0)
    assert not _check_if_channel_contains_timeseries(fns_xr, 1)
    assert not _check_if_channel_contains_timeseries(fns_xr, 2)
    assert not _check_if_channel_contains_timeseries(fns_xr, 3)


def test_get_z_spacing(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert _get_z_spacing(fns_xr) == 0

    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert _get_z_spacing(fns_xr) == 2.99

    with pytest.raises(ValueError):
        _get_z_spacing(fns_xr[:, :, :, 1:, :])


def test_get_t_spacing(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert _get_t_spacing(fns_xr) == 0

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "direct_transfer" / "3435"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert _get_t_spacing(fns_xr) == 30.029

    with pytest.raises(ValueError):
        _get_t_spacing(fns_xr[:, :, :, 1:, :])


def test_build_channel_metadata(temp_dir):
    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    output = _build_channel_metadata(fns_xr)
    assert list(output.keys()) == ["w1", "w3", "w4"]
    assert output["w1"]["plate_name"] == "9987"
    assert output["w1"]["channel_name"] == "DAPI"
    assert output["w1"]["dx"] == 1.3672
    assert output["w1"]["dy"] == 1.3672
    assert output["w1"]["dz"] == 2.99
    assert output["w1"]["spatial_calibration_units"] == "µm"
    assert output["w1"]["dt"] == 0
    assert output["w1"]["time_calibration_units"] == "s"
    assert output["w1"]["wavelength"] == 452.0
    assert output["w1"]["exposure_time"] == 100.0
    assert output["w1"]["exposure_time_unit"] == "ms"
    assert output["w1"]["objective"] == "40X Plan Apo Lambda"

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "direct_transfer" / "3435"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    output = _build_channel_metadata(fns_xr)
    assert list(output.keys()) == ["w1", "w2", "w3", "w4"]
    assert output["w1"]["plate_name"] == "9987"
    assert output["w1"]["channel_name"] == "DAPI"
    assert output["w1"]["dx"] == 1.3672
    assert output["w1"]["dy"] == 1.3672
    assert output["w1"]["dz"] == 0
    assert output["w1"]["spatial_calibration_units"] == "µm"
    assert output["w1"]["dt"] == 30.029
    assert output["w1"]["time_calibration_units"] == "s"
    assert output["w1"]["wavelength"] == 452.0
    assert output["w1"]["exposure_time"] == 100.0
    assert output["w1"]["exposure_time_unit"] == "ms"
    assert output["w1"]["objective"] == "40X Plan Apo Lambda"


def test_read_images(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)

    output = _read_images(fns_xr.data, ny=256, nx=256, im_dtype="uint16")
    assert output.shape == (1, 1, 1, 1, 1, 256, 256)

    output = _read_images(
        fns_xr.data[0, 0, 0, 0, :1], ny=256, nx=256, im_dtype="uint16"
    )
    assert output.shape == (1, 256, 256)
    assert any(np.unique(output) != 0)

    fns_xr.data[0, 0, 0, 0, 0] = ""
    output = _read_images(fns_xr, ny=256, nx=256, im_dtype="uint16")
    assert all(np.unique(output) == 0)


def test_read_stage_positions(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)

    output = _read_stage_positions(fns_xr.data)
    assert output.shape == (1, 1, 1, 1, 1, 3)

    output = _read_stage_positions(fns_xr.data[0, 0, 0, 0, :1])
    assert output.shape == (1, 3)
    npt.assert_array_equal(output, [[10025.4, 19885.0, 22985.0]])


def test_lazy_load_plate_as_xr(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    data_xr = lazy_load_images(fns_xr)
    assert data_xr.shape == (1, 1, 1, 1, 1, 256, 256)
    assert data_xr.compute().shape == (1, 1, 1, 1, 1, 256, 256)
    assert data_xr.dtype == "uint16"

    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    data_xr = lazy_load_images(fns_xr)
    assert data_xr.shape == (2, 2, 1, 3, 3, 256, 256)
    assert data_xr.compute().shape == (2, 2, 1, 3, 3, 256, 256)
    assert data_xr.dtype == "uint16"

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "direct_transfer" / "3435"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    data_xr = lazy_load_images(fns_xr)
    assert data_xr.shape == (2, 2, 6, 4, 1, 256, 256)
    assert data_xr.compute().shape == (2, 2, 6, 4, 1, 256, 256)


def test_lazy_load_stage_positions_as_xr(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    files_df = parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    positions_xr = lazy_load_stage_positions(fns_xr)
    assert positions_xr.shape == (1, 1, 1, 1, 1, 3)
    assert positions_xr.compute().shape == (1, 1, 1, 1, 1, 3)
    assert positions_xr.dtype == "float64"
    npt.assert_array_equal(positions_xr, [[[[[[10025.4, 19885.0, 22985.0]]]]]])
