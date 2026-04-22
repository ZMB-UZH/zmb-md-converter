"""Tests for MD ImageXpress HCS.ai parsing functions."""

import pytest

from zmb_md_converter.io.hcs_ai.parsing import (
    parse_jdce_metadata,
    parse_MD_plate_folder,
)


@pytest.fixture(scope="module")
def data_dir(hcs_ai_temp_dir):
    """Montage z-stack dataset:
    experiment/, experiment_z_stack/, experiment_montage/."""
    return (
        hcs_ai_temp_dir
        / "1w4s1t1c4z_binning4x4_tiled-L_montage-stitch-zstack"
        / "test_data_20260316_145602"
    )


@pytest.fixture(scope="module")
def data_dir_stack_only(hcs_ai_temp_dir):
    """Stack-only dataset:
    experiment_z_stack/ only, no experiment/ projection folder."""
    return (
        hcs_ai_temp_dir
        / "1w1s1t2c4z_binning4x4_stack-only"
        / "test_data_20260302_141940"
    )


@pytest.fixture(scope="module")
def data_dir_mip_only(hcs_ai_temp_dir):
    """MIP-only dataset: experiment/ only, no z-stack."""
    return (
        hcs_ai_temp_dir / "1w1s1t2c4z_binning4x4_MIP-only" / "test_data_20260302_141908"
    )


def test_parse_jdce_metadata(data_dir):
    jdce_path = (
        data_dir
        / "experiment_z_stack"
        / "1w4s1t1c4z_binning4x4_tiled-L_montage-stitch-zstack_z_stack.jdce"
    )
    meta = parse_jdce_metadata(jdce_path)

    assert meta["dx"] == pytest.approx(1.294)
    assert meta["dy"] == pytest.approx(1.294)
    assert meta["dz"] == pytest.approx(4.0)
    assert meta["spatial_calibration_units"] == "µm"
    assert len(meta["channels"]) == 1
    assert meta["channels"][0]["channel_name"] == "DAPI"


def test_parse_MD_plate_folder_returns_dataframe(data_dir):
    import pandas as pd

    df = parse_MD_plate_folder(data_dir)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "name",
        "well",
        "field",
        "channel",
        "time_point",
        "z",
        "path",
        "dx",
        "dy",
        "dz",
        "dt",
    }
    assert expected_cols.issubset(set(df.columns))


def test_parse_MD_plate_folder_wells(data_dir):
    df = parse_MD_plate_folder(data_dir)

    assert set(df["well"].unique()) == {"C05"}


def test_parse_MD_plate_folder_z_and_projections(data_dir):
    df = parse_MD_plate_folder(data_dir)

    # z-stack rows have string z values
    z_stack = df[df["z"].notna()]
    assert set(z_stack["z"].unique()) == {"0", "1", "2", "3"}

    # projection rows have z=NaN
    projections = df[df["z"].isna()]
    assert not projections.empty


def test_parse_MD_plate_folder_only_2D(data_dir):
    df = parse_MD_plate_folder(data_dir, only_2D=True)

    assert df is not None
    assert df["z"].isna().all()


def test_parse_MD_plate_folder_use_montage(data_dir):
    df = parse_MD_plate_folder(data_dir, use_montage=True)

    assert df is not None
    # montage z-stack rows should be present
    assert df["z"].notna().any()


def test_parse_MD_plate_folder_only_2D_with_montage_prefers_montage_data(data_dir):
    df = parse_MD_plate_folder(data_dir, only_2D=True, use_montage=True)

    assert df is not None
    # When montage is requested in only_2D mode, keep montage dimensionality
    # so projections can be generated later from stitched images.
    assert df["z"].notna().any()


def test_parse_MD_plate_folder_use_montage_not_available_falls_back(
    data_dir_stack_only, capsys
):
    """When montage folder is absent, falls back to regular z-stack."""
    df = parse_MD_plate_folder(data_dir_stack_only, use_montage=True)

    assert df is not None
    assert df["z"].notna().any()
    captured = capsys.readouterr()
    assert "experiment_montage" in captured.out


def test_parse_MD_plate_folder_only_2D_with_montage_not_available_falls_back(
    data_dir_stack_only, capsys
):
    """When montage and experiment/ are absent, falls back to z-stack (projections
    computed downstream)."""
    df = parse_MD_plate_folder(data_dir_stack_only, only_2D=True, use_montage=True)

    assert df is not None
    assert not df.empty
    captured = capsys.readouterr()
    assert "experiment_montage" in captured.out
    assert "experiment_z_stack" in captured.out or "z-stack" in captured.out


def test_parse_MD_plate_folder_stack_only(data_dir_stack_only):
    """experiment_z_stack/ only, no experiment/ projection folder."""
    df = parse_MD_plate_folder(data_dir_stack_only)

    assert df is not None
    assert df["z"].notna().any()
    assert df["z"].isna().sum() == 0


def test_parse_MD_plate_folder_mip_only(data_dir_mip_only):
    """experiment/ only (no z-stack), treated as projections."""
    df = parse_MD_plate_folder(data_dir_mip_only)

    assert df is not None
    assert df["z"].isna().all()


def test_parse_MD_plate_folder_returns_none_for_invalid_dir(tmp_path):
    result = parse_MD_plate_folder(tmp_path)
    assert result is None
