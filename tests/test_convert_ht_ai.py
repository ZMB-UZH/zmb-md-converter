"""Tests for convert_ht_ai_to_tiffs.

Datasets used (downloaded from Zenodo via the ``temp_dir`` fixture):
  direct_transfer/3420  – 1w 1s 1t 1ch, no z-stack
  direct_transfer/3433  – 2w 2s 1t 2ch 3z  (clean, uniform acquisition)
  direct_transfer/3434  – 2w 2s 1t 4ch, mixed z-sampling (w2=proj-only, w3/w4=1-plane)
  direct_transfer/3435  – 2w 2s 6t 4ch, mixed time-sampling (2D)
"""

from pathlib import Path

import pytest
import tifffile

from zmb_md_converter.convert import convert_ht_ai_to_tiffs


def _read_output_tiffs(out_dir: Path) -> list:
    """Return sorted list of (path, axes, shape) tuples for all .ome.tiff files."""
    files = sorted(out_dir.glob("*.ome.tiff"))
    results = []
    for f in files:
        with tifffile.TiffFile(f) as tif:
            series = tif.series[0]
            results.append((f, series.axes, series.shape))
    return results


def _dim_size(axes: str, shape: tuple, dim: str) -> int:
    """Return size of *dim* (e.g. 'Z', 'T', 'C'); 1 if absent (squeezed out)."""
    if dim in axes:
        return shape[axes.index(dim)]
    return 1


# ---------------------------------------------------------------------------
# Basic 3D conversion (only_2D=False)
# ---------------------------------------------------------------------------


def test_convert_ht_ai_3d(ht_ai_temp_dir, tmp_path):
    """3D stack is written with correct channel and z-plane count."""
    convert_ht_ai_to_tiffs(
        input_dir=ht_ai_temp_dir / "direct_transfer" / "3433",
        output_dir=tmp_path,
        only_2D=False,
        dimensions_to_split=("field",),
        fill_mixed_acquisition=False,
    )
    results = _read_output_tiffs(tmp_path)
    # 2 wells × 2 fields = 4 output files
    assert len(results) == 4
    _path, axes, shape = results[0]
    assert _dim_size(axes, shape, "Z") == 3  # 3 z-planes
    assert _dim_size(axes, shape, "C") == 2  # 2 channels


# ---------------------------------------------------------------------------
# 2D conversion via projections (only_2D=True)
# ---------------------------------------------------------------------------


def test_convert_ht_ai_2d(ht_ai_temp_dir, tmp_path):
    """Projection images are exported as single-plane stacks when only_2D=True."""
    convert_ht_ai_to_tiffs(
        input_dir=ht_ai_temp_dir / "direct_transfer" / "3433",
        output_dir=tmp_path,
        only_2D=True,
        dimensions_to_split=("field",),
        fill_mixed_acquisition=False,
    )
    results = _read_output_tiffs(tmp_path)
    assert len(results) == 4
    _path, axes, shape = results[0]
    assert _dim_size(axes, shape, "Z") == 1  # single projection plane
    assert _dim_size(axes, shape, "C") == 2


# ---------------------------------------------------------------------------
# Single-image dataset
# ---------------------------------------------------------------------------


def test_convert_ht_ai_single_image(ht_ai_temp_dir, tmp_path):
    """Single-image dataset (1w 1s 1t 1ch) produces exactly one output file."""
    convert_ht_ai_to_tiffs(
        input_dir=ht_ai_temp_dir / "direct_transfer" / "3420",
        output_dir=tmp_path,
        only_2D=False,
        dimensions_to_split=("field",),
    )
    results = _read_output_tiffs(tmp_path)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Mixed z-acquisition with fill
# ---------------------------------------------------------------------------


def test_convert_ht_ai_fill_mixed_acquisition(ht_ai_temp_dir, tmp_path):
    """Mixed z-sampling is filled to a uniform 3-plane stack for all 3D channels."""
    # 3434: w1 has z=1/2/3; w3, w4 have z=1 only; w2 is projection-only (dropped for 3D)
    convert_ht_ai_to_tiffs(
        input_dir=ht_ai_temp_dir / "direct_transfer" / "3434",
        output_dir=tmp_path,
        only_2D=False,
        dimensions_to_split=("field",),
        fill_mixed_acquisition=True,
    )
    results = _read_output_tiffs(tmp_path)
    # 2 wells × 2 fields = 4 output files
    assert len(results) == 4
    _path, axes, shape = results[0]
    # fill extends all 4 channels (including w2 which is projection-only) to 3 planes
    assert _dim_size(axes, shape, "C") == 4
    assert _dim_size(axes, shape, "Z") == 3


# ---------------------------------------------------------------------------
# Timeseries with mixed time-sampling
# ---------------------------------------------------------------------------


def test_convert_ht_ai_timeseries(ht_ai_temp_dir, tmp_path):
    """Timeseries is exported with all 6 timepoints after fill."""
    # 3435: w1 has t=1-6; others partial; fill extends all to 6 timepoints
    convert_ht_ai_to_tiffs(
        input_dir=ht_ai_temp_dir / "direct_transfer" / "3435",
        output_dir=tmp_path,
        only_2D=True,
        dimensions_to_split=("field",),
        fill_mixed_acquisition=True,
    )
    results = _read_output_tiffs(tmp_path)
    # 2 wells × 2 fields = 4 output files
    assert len(results) == 4
    _path, axes, shape = results[0]
    assert _dim_size(axes, shape, "T") == 6  # 6 timepoints after fill
    assert _dim_size(axes, shape, "C") == 4  # 4 channels


# ---------------------------------------------------------------------------
# Invalid directory → ValueError
# ---------------------------------------------------------------------------


def test_convert_ht_ai_invalid_dir(tmp_path):
    """A directory without MD data raises ValueError."""
    with pytest.raises(ValueError, match="No files found"):
        convert_ht_ai_to_tiffs(tmp_path, tmp_path)
