import os

import numpy.testing as npt
import pytest

from zmb_md_converter.parsing import (
    _fill_mixed_acquisitions,
    _parse_file,
    _parse_MD_plate_folder,
    _parse_MD_tz_folder,
)


@pytest.mark.parametrize(
    "input_fn,expected_dict",
    [
        (
            os.path.join("TimePoint_1", "ZStep_0", "1t-3w-4s-2c-5z_B02_s1_w1.TIF"),
            {
                "time_point": "1",
                "z": "0",
                "name": "1t-3w-4s-2c-5z",
                "well": "B02",
                "field": "s1",
                "channel": "w1",
                "md_id": None,
                "ext": ".TIF",
            },
        ),
        (
            os.path.join(
                "TimePoint_1", "9987_B02_s1_w1F121202F-242E-4F66-9AE6-1324A19D144D.tif"
            ),
            {
                "time_point": "1",
                "z": None,
                "name": "9987",
                "well": "B02",
                "field": "s1",
                "channel": "w1",
                "md_id": "F121202F-242E-4F66-9AE6-1324A19D144D",
                "ext": ".tif",
            },
        ),
        (
            os.path.join(
                "TimePoint_1",
                "ZStep_5",
                "9987_C02_s6_w10AF09CDA-6C27-4635-8BB8-F4F6EB03E088.tif",
            ),
            {
                "time_point": "1",
                "z": "5",
                "name": "9987",
                "well": "C02",
                "field": "s6",
                "channel": "w1",
                "md_id": "0AF09CDA-6C27-4635-8BB8-F4F6EB03E088",
                "ext": ".tif",
            },
        ),
    ],
)
def test_parse_file(input_fn, expected_dict):
    result = _parse_file(input_fn)
    assert result == expected_dict


def test_parse_MD_plate_folder_directTransfer(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "direct_transfer" / "3420"
    df = _parse_MD_plate_folder(root_dir)
    df.sort_values(by=["path"], inplace=True)
    assert len(df) == 1
    npt.assert_array_equal(df["time_point"].unique(), ["1"])
    npt.assert_array_equal(df["z"].unique(), [None])
    npt.assert_array_equal(df["name"].unique(), ["9987"])
    npt.assert_array_equal(df["well"].unique(), ["B02"])
    npt.assert_array_equal(df["field"].unique(), ["s1"])
    npt.assert_array_equal(df["channel"].unique(), ["w0"])
    npt.assert_array_equal(df["dir_name"].unique(), ["3420"])

    # 1t-3z-2w-2s-2c
    root_dir = temp_dir / "direct_transfer" / "3433"
    df = _parse_MD_plate_folder(root_dir)
    df.sort_values(by=["path"], inplace=True)
    assert len(df) == 32
    npt.assert_array_equal(df["time_point"].unique(), ["1"])
    npt.assert_array_equal(df["z"].unique(), [None, "1", "2", "3"])
    npt.assert_array_equal(df["name"].unique(), ["9987"])
    npt.assert_array_equal(df["well"].unique(), ["B02", "B03"])
    npt.assert_array_equal(df["field"].unique(), ["s1", "s2"])
    npt.assert_array_equal(df["channel"].unique(), ["w1", "w2"])
    npt.assert_array_equal(df["dir_name"].unique(), ["3433"])

    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "direct_transfer" / "3434"
    df = _parse_MD_plate_folder(root_dir)
    df.sort_values(by=["path"], inplace=True)
    assert len(df) == 28
    npt.assert_array_equal(df["time_point"].unique(), ["1"])
    npt.assert_array_equal(df["z"].unique(), [None, "1", "2", "3"])
    npt.assert_array_equal(df["name"].unique(), ["9987"])
    npt.assert_array_equal(df["well"].unique(), ["B02", "B03"])
    npt.assert_array_equal(df["field"].unique(), ["s1", "s2"])
    npt.assert_array_equal(df["channel"].unique(), ["w1", "w2", "w3", "w4"])
    npt.assert_array_equal(df["dir_name"].unique(), ["3434"])
    # all z-planes:
    npt.assert_array_equal(
        df.query("channel=='w1'")["z"].unique(), [None, "1", "2", "3"]
    )
    # only projection:
    npt.assert_array_equal(df.query("channel=='w2'")["z"].unique(), [None])
    # only 1 plane (0um offset):
    npt.assert_array_equal(df.query("channel=='w3'")["z"].unique(), ["1"])
    # only 1 plane (10um offset):
    npt.assert_array_equal(df.query("channel=='w4'")["z"].unique(), ["1"])

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "direct_transfer" / "3435"
    df = _parse_MD_plate_folder(root_dir)
    df.sort_values(by=["path"], inplace=True)
    assert len(df) == 44
    npt.assert_array_equal(df["time_point"].unique(), ["1", "2", "3", "4", "5", "6"])
    npt.assert_array_equal(df["z"].unique(), [None])
    npt.assert_array_equal(df["name"].unique(), ["9987"])
    npt.assert_array_equal(df["well"].unique(), ["B02", "B03"])
    npt.assert_array_equal(df["field"].unique(), ["s1", "s2"])
    npt.assert_array_equal(df["channel"].unique(), ["w1", "w2", "w3", "w4"])
    npt.assert_array_equal(df["dir_name"].unique(), ["3435"])
    # all timepoints:
    npt.assert_array_equal(
        df.query("channel=='w1'")["time_point"].unique(), ["1", "2", "3", "4", "5", "6"]
    )
    # at first timepoint:
    npt.assert_array_equal(df.query("channel=='w2'")["time_point"].unique(), ["1"])
    # at first and last timepoint:
    npt.assert_array_equal(df.query("channel=='w3'")["time_point"].unique(), ["1", "6"])
    # at every 3rd timepoint:
    npt.assert_array_equal(df.query("channel=='w4'")["time_point"].unique(), ["1", "4"])

    # test wrong folder
    root_dir = temp_dir / "direct_transfer"
    df = _parse_MD_plate_folder(root_dir)
    assert df is None


def test_parse_MD_plate_folder_MetaXpress(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3420"
    df = _parse_MD_plate_folder(root_dir)
    df.sort_values(by=["path"], inplace=True)
    assert len(df) == 1
    npt.assert_array_equal(df["time_point"].unique(), ["1"])
    npt.assert_array_equal(df["z"].unique(), [None])
    npt.assert_array_equal(df["name"].unique(), ["9987"])
    npt.assert_array_equal(df["well"].unique(), ["B02"])
    npt.assert_array_equal(df["field"].unique(), [None])
    npt.assert_array_equal(df["channel"].unique(), ["w0"])
    npt.assert_array_equal(df["dir_name"].unique(), ["9987_Plate_3420"])

    # 1t-3z-2w-2s-2c
    root_dir = temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3433"
    df = _parse_MD_plate_folder(root_dir)
    df.sort_values(by=["path"], inplace=True)
    assert len(df) == 32
    npt.assert_array_equal(df["time_point"].unique(), ["1"])
    npt.assert_array_equal(df["z"].unique(), [None, "1", "2", "3"])
    npt.assert_array_equal(df["name"].unique(), ["9987"])
    npt.assert_array_equal(df["well"].unique(), ["B02", "B03"])
    npt.assert_array_equal(df["field"].unique(), ["s1", "s2"])
    npt.assert_array_equal(df["channel"].unique(), ["w1", "w2"])
    npt.assert_array_equal(df["dir_name"].unique(), ["9987_Plate_3433"])

    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3434"
    df = _parse_MD_plate_folder(root_dir)
    df.sort_values(by=["path"], inplace=True)
    assert len(df) == 64
    npt.assert_array_equal(df["time_point"].unique(), ["1"])
    npt.assert_array_equal(df["z"].unique(), [None, "1", "2", "3"])
    npt.assert_array_equal(df["name"].unique(), ["9987"])
    npt.assert_array_equal(df["well"].unique(), ["B02", "B03"])
    npt.assert_array_equal(df["field"].unique(), ["s1", "s2"])
    npt.assert_array_equal(df["channel"].unique(), ["w1", "w2", "w3", "w4"])
    npt.assert_array_equal(df["dir_name"].unique(), ["9987_Plate_3434"])
    # all z-planes:
    npt.assert_array_equal(
        df.query("channel=='w1'")["z"].unique(), [None, "1", "2", "3"]
    )
    # only projection:
    npt.assert_array_equal(
        df.query("channel=='w2'")["z"].unique(), [None, "1", "2", "3"]
    )
    # only 1 plane (0um offset):
    npt.assert_array_equal(
        df.query("channel=='w3'")["z"].unique(), [None, "1", "2", "3"]
    )
    # only 1 plane (10um offset):
    npt.assert_array_equal(
        df.query("channel=='w4'")["z"].unique(), [None, "1", "2", "3"]
    )

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3435"
    df = _parse_MD_plate_folder(root_dir)
    df.sort_values(by=["path"], inplace=True)
    assert len(df) == 96
    npt.assert_array_equal(df["time_point"].unique(), ["1", "2", "3", "4", "5", "6"])
    npt.assert_array_equal(df["z"].unique(), [None])
    npt.assert_array_equal(df["name"].unique(), ["9987"])
    npt.assert_array_equal(df["well"].unique(), ["B02", "B03"])
    npt.assert_array_equal(df["field"].unique(), ["s1", "s2"])
    npt.assert_array_equal(df["channel"].unique(), ["w1", "w2", "w3", "w4"])
    npt.assert_array_equal(df["dir_name"].unique(), ["9987_Plate_3435"])
    # all timepoints:
    npt.assert_array_equal(
        df.query("channel=='w1'")["time_point"].unique(), ["1", "2", "3", "4", "5", "6"]
    )
    # at first timepoint:
    npt.assert_array_equal(
        df.query("channel=='w2'")["time_point"].unique(), ["1", "2", "3", "4", "5", "6"]
    )
    # at first and last timepoint:
    npt.assert_array_equal(
        df.query("channel=='w3'")["time_point"].unique(), ["1", "2", "3", "4", "5", "6"]
    )
    # at every 3rd timepoint:
    npt.assert_array_equal(
        df.query("channel=='w4'")["time_point"].unique(), ["1", "2", "3", "4", "5", "6"]
    )


def test_parse_MD_tz_folder(temp_dir):
    # 1t-1z-1w-1s-1c
    root_dir = temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3420"
    df = _parse_MD_tz_folder(root_dir)
    assert df is None

    # timeseries
    root_dir = temp_dir / "direct_transfer" / "timeseries"
    df = _parse_MD_tz_folder(root_dir)
    assert len(df) == 96
    npt.assert_array_equal(df.time_point.unique(), ["1", "2", "3"])

    # timeseries
    root_dir = temp_dir / "MetaXpress_all-z_include-projection" / "timeseries"
    df = _parse_MD_tz_folder(root_dir)
    assert len(df) == 96
    npt.assert_array_equal(df.time_point.unique(), ["1", "2", "3"])

    # folder with random plates
    root_dir = temp_dir / "MetaXpress_all-z_include-projection"
    with pytest.raises(ValueError):
        _parse_MD_tz_folder(root_dir)


def test_fill_mixed_acquisitions(temp_dir):
    # 1t-3z-2w-2s-4c mixed z-sampliing
    root_dir = temp_dir / "direct_transfer" / "3434"
    df = _parse_MD_plate_folder(root_dir)
    df = _fill_mixed_acquisitions(df)
    assert len(df) == 64
    for c in df.channel.unique():
        assert len(df[df.channel == c].z.unique()) == len(df.z.unique())

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "direct_transfer" / "3435"
    df = _parse_MD_plate_folder(root_dir)
    df = _fill_mixed_acquisitions(df)
    assert len(df) == 96
    for c in df.channel.unique():
        assert len(df[df.channel == c].time_point.unique()) == len(
            df.time_point.unique()
        )

    # 6t-1z-2w-2s-4c mixed time-sampling
    root_dir = temp_dir / "direct_transfer" / "3435"
    df = _parse_MD_plate_folder(root_dir)
    df = df[~((df.time_point == "1") & (df.channel == "w1"))]
    with pytest.raises(ValueError):
        _fill_mixed_acquisitions(df)

    assert _fill_mixed_acquisitions(None) is None
