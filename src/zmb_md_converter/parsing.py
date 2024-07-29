import glob
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

_file_pattern = re.compile(
    r"(TimePoint_(?P<time_point>\d+))/"
    r"(ZStep_(?P<z>\d+)/)?"
    r"(?P<name>[^_]*)_(?P<well>[A-Z]+\d{2})"
    r"(_(?P<field>s\d+))?"
    r"(_(?P<channel>w[1-9]{1}))?"
    r"(?!_thumb)"
    r"(?P<md_id>[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12})?"
    r"(?P<ext>.tif|.TIF)"
)


def _parse_file(fn: str) -> dict:
    fn = Path(fn).as_posix()
    m = _file_pattern.fullmatch(fn)
    if m:
        return m.groupdict()
    else:
        return {}


def _parse_MD_plate_folder(root_dir: Union[Path, str]) -> pd.DataFrame:
    fns1 = glob.glob("**/*.tif", root_dir=root_dir, recursive=True)
    fns2 = glob.glob("**/*.TIF", root_dir=root_dir, recursive=True)
    fns = fns1 + fns2
    files = []
    for fn in fns:
        row = _parse_file(fn)
        if row:
            row["path"] = str(Path(root_dir).joinpath(fn))
            row["dir_name"] = Path(root_dir).name
            files.append(row)

    if files:
        df = pd.DataFrame(files)
        df.loc[df.z == "0", "z"] = None
        return df
    else:
        return None


def _fill_mixed_acquisitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to fill missing slices and timepoints in a dataframe.

    If the data was transferred directly and it was acquired in a mixed way
    (e.g. some channels have only projections, or only one timepoint), some
    slices are missing. This function fills the missing slices with the
    available data. !!!This will in the end duplicate some data and might not
    be the best solution!!!
    """
    if df is None:
        return None

    df = df.copy()

    for c in df.channel.unique():
        # HANDLE SLICES
        # case 1: only projection is saved
        # -> fill all z with projection
        if (
            df[df.channel == c].z.unique()
            == [
                None,
            ]
        ).all():
            sub_df = df[(df.channel == c) & df.z.isna()].copy()
            for z in df.z.unique():
                if z is not None:
                    sub_df.z = z
                    df = pd.concat([df, sub_df], ignore_index=True)
        # case 2: only one z is saved
        # -> fill all z with that slice
        if (
            df[df.channel == c].z.unique()
            == [
                "1",
            ]
        ).all():
            sub_df = df[(df.channel == c) & (df.z == "1")].copy()
            for z in df.z.unique():
                if z != "1":
                    sub_df.z = z
                    df = pd.concat([df, sub_df], ignore_index=True)

        # HANDLE TIMEPOINTS
        # case 1: only first timepoint is saved
        # -> fill all timepoints with first timepoint
        # case 2: only first and last timepoint is saved
        # -> fill remaining timepoints with first timepoint
        # case 3: only every nth timepoint is saved
        # -> fill remaining timepoints with latest timepoint
        all_tps = df.time_point.unique()
        all_tps = np.sort([int(t) for t in all_tps])
        if len(df[df.channel == c].time_point.unique()) != len(all_tps):
            if str(all_tps[0]) not in df[df.channel == c].time_point.unique():
                raise ValueError(f"First timepoint is missing for channel {c}")
            for tp in all_tps:
                if str(tp) in df[df.channel == c].time_point.unique():
                    sub_df = df[(df.channel == c) & (df.time_point == str(tp))].copy()
                else:
                    sub_df.time_point = str(tp)
                    df = pd.concat([df, sub_df], ignore_index=True)
    return df
