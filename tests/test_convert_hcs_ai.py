"""Tests for convert_hcs_ai_to_tiffs."""

from pathlib import Path

import pytest
import tifffile

from zmb_md_converter.convert import convert_hcs_ai_to_tiffs


def _read_output_tiffs(out_dir: Path) -> list:
    """Return sorted list of (path, axes, shape) tuples for all .ome.tiff files."""
    files = sorted(out_dir.glob("*.ome.tiff"))
    results = []
    for f in files:
        with tifffile.TiffFile(f) as tif:
            series = tif.series[0]
            results.append((f, series.axes, series.shape))
    return results


def _z_size(axes: str, shape: tuple) -> int:
    """Return size of Z dimension; 1 if Z is absent (squeezed out)."""
    if "Z" in axes:
        return shape[axes.index("Z")]
    return 1


# ---------------------------------------------------------------------------
# Basic 3D conversion (only_2D=False, use_montage=False)
# ---------------------------------------------------------------------------


def test_convert_hcs_ai_3d(hcs_ai_temp_dir, tmp_path):
    """3D stack is written with the correct number of z-planes."""
    data_dir = (
        hcs_ai_temp_dir
        / "1w1s1t2c4z_binning4x4_stack-and-MIP"
        / "test_data_20260302_141751"
    )
    convert_hcs_ai_to_tiffs(
        data_dir,
        tmp_path,
        only_2D=False,
        use_montage=False,
        dimensions_to_split=("field",),
    )
    results = _read_output_tiffs(tmp_path)
    # 1 well x 1 field = 1 output file
    assert len(results) == 1
    _path, axes, shape = results[0]
    assert _z_size(axes, shape) == 4  # Z = 4 planes


# ---------------------------------------------------------------------------
# Basic 2D conversion via pre-calculated MIP (only_2D=True, use_montage=False)
# ---------------------------------------------------------------------------


def test_convert_hcs_ai_2d_from_mip(hcs_ai_temp_dir, tmp_path):
    """Pre-calculated MIP is used when only_2D=True and MIP folder exists."""
    data_dir = (
        hcs_ai_temp_dir
        / "1w1s1t2c4z_binning4x4_stack-and-MIP"
        / "test_data_20260302_141751"
    )
    convert_hcs_ai_to_tiffs(
        data_dir,
        tmp_path,
        only_2D=True,
        use_montage=False,
        dimensions_to_split=("field",),
    )
    results = _read_output_tiffs(tmp_path)
    assert len(results) == 1
    _path, axes, shape = results[0]
    assert _z_size(axes, shape) == 1  # Z = 1 (projection)


# ---------------------------------------------------------------------------
# Fallback: only z-stack present, only_2D=True → auto-project
# ---------------------------------------------------------------------------


def test_convert_hcs_ai_2d_from_stack_fallback(hcs_ai_temp_dir, tmp_path):
    """When no MIP folder exists, max-projection is computed from the stack."""
    data_dir = (
        hcs_ai_temp_dir
        / "1w1s1t2c4z_binning4x4_stack-only"
        / "test_data_20260302_141940"
    )
    convert_hcs_ai_to_tiffs(
        data_dir,
        tmp_path,
        only_2D=True,
        use_montage=False,
        dimensions_to_split=("field",),
    )
    results = _read_output_tiffs(tmp_path)
    assert len(results) == 1
    _path, axes, shape = results[0]
    assert _z_size(axes, shape) == 1  # Z = 1 (computed projection)


# ---------------------------------------------------------------------------
# Multi-well: correct output file count
# ---------------------------------------------------------------------------


def test_convert_hcs_ai_multi_well(hcs_ai_temp_dir, tmp_path):
    """Two wells x four fields → eight output files."""
    data_dir = hcs_ai_temp_dir / "2w4s3t2c4z_binning4x4" / "test_data_20260302_143452"
    convert_hcs_ai_to_tiffs(
        data_dir,
        tmp_path,
        only_2D=False,
        use_montage=False,
        dimensions_to_split=("field",),
    )
    results = _read_output_tiffs(tmp_path)
    assert len(results) == 8  # 2 wells x 4 fields


# ---------------------------------------------------------------------------
# Montage z-stack + only_2D=True → project from stitched stack
# ---------------------------------------------------------------------------


def test_convert_hcs_ai_montage_zstack_2d(hcs_ai_temp_dir, tmp_path):
    """With use_montage=True + only_2D=True the stitched stack is projected."""
    data_dir = (
        hcs_ai_temp_dir
        / "1w4s1t1c4z_binning4x4_tiled-L_montage-stitch-zstack"
        / "test_data_20260316_145602"
    )
    convert_hcs_ai_to_tiffs(
        data_dir,
        tmp_path,
        only_2D=True,
        use_montage=True,
        dimensions_to_split=("field",),
    )
    results = _read_output_tiffs(tmp_path)
    assert len(results) >= 1
    _path, axes, shape = results[0]
    assert _z_size(axes, shape) == 1  # Z = 1 (computed projection from montage stack)


# ---------------------------------------------------------------------------
# Invalid directory → ValueError
# ---------------------------------------------------------------------------


def test_convert_hcs_ai_invalid_dir(tmp_path):
    """A directory without a .mxprotocol file raises ValueError."""
    with pytest.raises(ValueError, match="No HCS.ai acquisition found"):
        convert_hcs_ai_to_tiffs(tmp_path, tmp_path)
