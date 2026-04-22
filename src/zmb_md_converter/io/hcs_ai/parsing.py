"""
Parsing functions for MD ImageXpress HCS.ai microscope data.

Folder structure
----------------
An HCS.ai acquisition folder is identified by the presence of a
``.mxprotocol`` file and may contain the following subfolders:

    <acquisition_name>/
        <acquisition_name>.mxprotocol        ← identifies an HCS.ai folder
        experiment/                           ← always 2D (MIP / projection /
            <name>.jdce                         single-plane)
            image_metadata_1.csv
            timepoint0/
                *.tif
        experiment_z_stack/                   ← 3D z-stack (if acquired)
            <name>_z_stack.jdce
            image_metadata_1.csv
            timepoint0/
                *.tif
        experiment_montage/                   ← stitched images (if requested
            <name>_montage.jdce                 during acquisition; may be 2D
            image_metadata_1.csv               or 3D depending on settings)
            timepoint0/
                *.tif

Combinations that occur in practice
-------------------------------------
- ``experiment/`` only → single-plane or MIP (no z-stack acquired)
- ``experiment_z_stack/`` only → z-stack, no projection saved
- Both → z-stack + MIP
- ``experiment_montage/`` in addition → stitched version of the above

The CSV metadata file contains per-image records with columns including
Row, Column, Field, Wavelength, Timepoint, ZIndex, ImageSubFolderPath,
ImageFileName, PositionXUm, PositionYUm, PositionZUm, etc.
"""

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

_STANDARD_ROW_NAMES = "ABCDEFGHIJKLMNOP"


def _row_col_to_well(row: int, col: int) -> str:
    """Convert 1-based row/column indices to well ID (e.g. row=3, col=5 → 'C05')."""
    return f"{_STANDARD_ROW_NAMES[row - 1]}{col:02d}"


def parse_jdce_metadata(jdce_path: Union[Path, str]) -> dict:
    """Parse a JDCE metadata file and return a flat metadata dictionary.

    Args:
        jdce_path: Path to the .jdce JSON file.

    Returns:
        dict with keys:
            name, dx, dy, dz, dt, spatial_calibration_units,
            time_calibration_units, channels (list of dicts with
            keys 'index', 'channel_name').
    """
    with open(jdce_path, encoding="utf-8") as f:
        data = json.load(f)

    protocol = data["ImageStack"]["AutoLeadAcquisitionProtocol"]
    calib = protocol["ObjectiveCalibration"]
    z_params = protocol["PlateMap"]["ZDimensionParameters"]
    time_schedule = protocol["PlateMap"]["TimeSchedule"]

    dz = z_params["Step"]
    if dz == 0.0:
        dz = 1.0

    times = time_schedule["Times"]
    dt = times[1]["Ms"] - times[0]["Ms"] if len(times) > 1 else 0.0

    channels = [
        {
            "index": wl["Index"],
            "channel_name": wl["EmissionFilter"]["Name"],
        }
        for wl in protocol["Wavelengths"]
    ]

    creation = data["ImageStack"]["Creation"]
    created_suffix = f" {creation['Date']}, {creation['Time']}"
    plate_id = data["ImageStack"]["PlateId"]
    name = plate_id.removesuffix(created_suffix)

    return {
        "name": name,
        "dx": calib["PixelWidth"],
        "dy": calib["PixelHeight"],
        "dz": dz,
        "dt": dt,
        "spatial_calibration_units": calib["Unit"],
        "time_calibration_units": "ms",
        "channels": channels,
    }


def _parse_experiment_dir(
    experiment_dir: Path,
    jdce_meta: dict,
    is_projection: bool,
) -> pd.DataFrame:
    """Parse a single experiment subfolder into a DataFrame.

    Args:
        experiment_dir: Path to the experiment subfolder.
        jdce_meta: Metadata dict returned by parse_jdce_metadata.
        is_projection: If True, z is set to NaN (projections/2D).
            If False, z is set to the ZIndex string from the CSV.

    Returns:
        DataFrame with columns: name, well, field, channel, time_point,
            z, path, dx, dy, dz, dt, spatial_calibration_units,
            time_calibration_units, channel_name.
    """
    csv_files = sorted(experiment_dir.glob("image_metadata_*.csv"))
    if not csv_files:
        return pd.DataFrame()

    raw_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    channel_name_map = {c["index"]: c["channel_name"] for c in jdce_meta["channels"]}

    rows = []
    for _, rec in raw_df.iterrows():
        well = _row_col_to_well(int(rec["Row"]), int(rec["Column"]))
        path = str(experiment_dir / rec["ImageSubFolderPath"] / rec["ImageFileName"])
        rows.append(
            {
                "name": jdce_meta["name"],
                "well": well,
                "field": f"s{int(rec['Field'])}",
                "channel": f"w{int(rec['Wavelength'])}",
                "time_point": str(int(rec["Timepoint"])),
                "z": None if is_projection else str(int(rec["ZIndex"])),
                "path": path,
                "dx": jdce_meta["dx"],
                "dy": jdce_meta["dy"],
                "dz": jdce_meta["dz"],
                "dt": jdce_meta["dt"],
                "spatial_calibration_units": jdce_meta["spatial_calibration_units"],
                "time_calibration_units": jdce_meta["time_calibration_units"],
                "channel_name": channel_name_map.get(int(rec["Wavelength"]), ""),
                "pos_x": float(rec["PositionXUm"]) if "PositionXUm" in rec and pd.notna(rec["PositionXUm"]) else 0.0,
                "pos_y": float(rec["PositionYUm"]) if "PositionYUm" in rec and pd.notna(rec["PositionYUm"]) else 0.0,
                "pos_z": float(rec["PositionZUm"]) if "PositionZUm" in rec and pd.notna(rec["PositionZUm"]) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _load_experiment_dir(experiment_dir: Path, is_projection: bool) -> pd.DataFrame:
    """Load a single experiment subfolder, returning an empty DataFrame on failure."""
    jdce_files = sorted(experiment_dir.glob("*.jdce"))
    if not jdce_files:
        return pd.DataFrame()
    jdce_meta = parse_jdce_metadata(jdce_files[0])
    return _parse_experiment_dir(experiment_dir, jdce_meta, is_projection=is_projection)


def parse_MD_plate_folder(
    root_dir: Union[Path, str],
    only_2D: bool = False,
    use_montage: bool = False,
) -> Optional[pd.DataFrame]:
    """Parse an MD ImageXpress HCS.ai acquisition folder.

    The folder is identified by the presence of a .mxprotocol file.

    Folder selection logic:

        - ``only_2D=True, use_montage=False``: prefer ``experiment/``
            (treated as projections, z=NaN). If ``experiment/`` is absent but
            ``experiment_z_stack/`` exists, the z-stack is loaded instead so
            that projections can be computed downstream.

        - ``only_2D=True, use_montage=True``: prefer ``experiment_montage/``
            and keep its native dimensionality (often z-stack). This allows later
            projection from stitched images. Falls back to ``experiment_z_stack/``
            (or ``experiment/``) with a printed note when no montage folder is
            present.

    - ``only_2D=False, use_montage=False`` (default): prefer
      ``experiment_z_stack/`` for 3D data; fall back to ``experiment/`` if
      no z-stack folder is present (covers acquisitions where only a single
      plane was acquired).  If a z-stack folder is found, the ``experiment/``
      projections are also included (z=NaN rows).

    - ``only_2D=False, use_montage=True``: use ``experiment_montage/``
      instead of ``experiment_z_stack/`` for 3D data.  Falls back to
      ``experiment_z_stack/`` with a printed note if no montage folder is
      found.  The ``experiment/`` projections are still included when
      available.

    Args:
        root_dir: Path to the acquisition root folder (the one containing
            the .mxprotocol file and experiment subfolders).
        only_2D: If True, read 2D projections from ``experiment/`` unless
            ``use_montage=True`` (then ``experiment_montage/`` is preferred).
        use_montage: If True, use stitched images from
            ``experiment_montage/`` instead of ``experiment_z_stack/``.

    Returns:
        DataFrame with image records, or None if root_dir is not a valid
        HCS.ai acquisition folder.
    """
    root_dir = Path(root_dir)

    if not any(root_dir.glob("*.mxprotocol")):
        return None

    subdirs = {d.name: d for d in root_dir.iterdir() if d.is_dir()}

    proj_dir = subdirs.get("experiment")
    z_stack_dir = subdirs.get("experiment_z_stack")
    montage_dir = subdirs.get("experiment_montage")

    dfs = []

    if only_2D:
        if use_montage and montage_dir is not None:
            # Keep montage data as stored (often z-stack) so projections can
            # be generated from stitched images later in the pipeline.
            df = _load_experiment_dir(montage_dir, is_projection=False)
            if not df.empty:
                dfs.append(df)
        else:
            if use_montage and montage_dir is None:
                print(
                    f"Note: No 'experiment_montage' folder found in {root_dir}. "
                    "Falling back to regular projections."
                )
            if proj_dir is not None:
                df = _load_experiment_dir(proj_dir, is_projection=True)
                if not df.empty:
                    dfs.append(df)
            elif z_stack_dir is not None:
                print(
                    f"Note: No 'experiment' folder found in {root_dir}. "
                    "Falling back to z-stack; projections will be computed downstream."
                )
                df = _load_experiment_dir(z_stack_dir, is_projection=False)
                if not df.empty:
                    dfs.append(df)
            else:
                raise FileNotFoundError(
                    f"No 'experiment' or 'experiment_z_stack' folder found in {root_dir}. "
                    "Cannot load 2D data."
                )

    else:
        # Select which folder provides the 3D (or fallback) data
        if use_montage:
            if montage_dir is None:
                print(
                    f"Note: No 'experiment_montage' folder found in {root_dir}. "
                    "Falling back to the regular z-stack."
                )
                stack_source = z_stack_dir
            else:
                stack_source = montage_dir
        elif z_stack_dir is not None:
            stack_source = z_stack_dir
        else:
            # No z-stack at all: fall back to experiment/ as single-plane data
            stack_source = None

        if stack_source is not None:
            df = _load_experiment_dir(stack_source, is_projection=False)
            if not df.empty:
                dfs.append(df)
            # Also include projections when a dedicated z-stack folder exists
            if proj_dir is not None:
                df_proj = _load_experiment_dir(proj_dir, is_projection=True)
                if not df_proj.empty:
                    dfs.append(df_proj)
        else:
            # Only experiment/ available (single-plane): treat as projection
            if proj_dir is not None:
                df = _load_experiment_dir(proj_dir, is_projection=True)
                if not df.empty:
                    dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)

