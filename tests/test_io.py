import pandas as pd
import pytest

from zmb_md_converter.io import create_filename_structure_MD
from zmb_md_converter.parsing import _parse_MD_plate_folder


def test_create_filename_structure_MD(temp_dir):
    # 1t-3z-2w-2s-2c
    root_dir = temp_dir / "direct_transfer" / "3433"
    files_df = _parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert fns_xr.shape == (2, 2, 1, 2, 3)
    assert fns_xr.well.values.tolist() == ["B02", "B03"]
    assert fns_xr.field.values.tolist() == ["s1", "s2"]
    assert fns_xr.time.values.tolist() == ["1"]
    assert fns_xr.channel.values.tolist() == ["w1", "w2"]
    assert fns_xr.plane.values.tolist() == ["1", "2", "3"]
    assert "" not in fns_xr

    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = _parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert fns_xr.shape == (2, 2, 1, 3, 3)
    assert fns_xr.well.values.tolist() == ["B02", "B03"]
    assert fns_xr.field.values.tolist() == ["s1", "s2"]
    assert fns_xr.time.values.tolist() == ["1"]
    assert fns_xr.channel.values.tolist() == ["w1", "w3", "w4"]
    assert fns_xr.plane.values.tolist() == ["1", "2", "3"]
    assert "" not in fns_xr[0, 0, 0, 0]
    assert fns_xr[0, 0, 0, 1, 0] != ""
    assert all(fns_xr[0, 0, 0, 1, 1:] == ["", ""])

    # 1t-3z-2w-2s-4c mixed z-sampliing - 3D
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = _parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df, only_2D=True)
    assert fns_xr.shape == (2, 2, 1, 2, 1)
    assert fns_xr.well.values.tolist() == ["B02", "B03"]
    assert fns_xr.field.values.tolist() == ["s1", "s2"]
    assert fns_xr.time.values.tolist() == ["1"]
    assert fns_xr.channel.values.tolist() == ["w1", "w2"]
    assert fns_xr.plane.values.tolist() == ["0"]
    assert "" not in fns_xr

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "direct_transfer" / "3435"
    files_df = _parse_MD_plate_folder(root_dir)
    fns_xr = create_filename_structure_MD(files_df)
    assert fns_xr.shape == (2, 2, 6, 4, 1)
    assert fns_xr.well.values.tolist() == ["B02", "B03"]
    assert fns_xr.field.values.tolist() == ["s1", "s2"]
    assert fns_xr.time.values.tolist() == ["1", "2", "3", "4", "5", "6"]
    assert fns_xr.channel.values.tolist() == ["w1", "w2", "w3", "w4"]
    assert fns_xr.plane.values.tolist() == ["0"]
    assert "" not in fns_xr[0, 0, :, 0, 0]
    assert fns_xr[0, 0, 0, 1, 0] != ""
    assert all(fns_xr[0, 0, 1:, 1, 0] == ["", "", "", "", ""])

    # check for 'no projections found' error:
    with pytest.raises(ValueError):
        create_filename_structure_MD(files_df[files_df.z.notnull()], only_2D=True)

    # check for 'multiple files found' error:
    root_dir = temp_dir / "direct_transfer" / "3434"
    files_df = _parse_MD_plate_folder(root_dir)
    with pytest.raises(RuntimeError):
        create_filename_structure_MD(pd.concat([files_df, files_df]))
