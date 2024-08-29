import itertools
import os

from zmb_md_converter.convert import convert_md_to_ome_tiffs


def test_convert_md_to_ome_tiffs(temp_dir):
    """
    Extensive test, testing all possible combinations. -> Takes ~ 2min.
    """
    root_dir_list = [
        temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3420",
        temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3433",
        temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3434",
        temp_dir / "MetaXpress_all-z_include-projection" / "9987_Plate_3435",
        temp_dir / "MetaXpress_all-z_include-projection" / "timeseries",
        temp_dir / "direct_transfer" / "3420",
        temp_dir / "direct_transfer" / "3433",
        temp_dir / "direct_transfer" / "3434",
        temp_dir / "direct_transfer" / "3435",
        temp_dir / "direct_transfer" / "timeseries",
    ]

    os.makedirs(temp_dir / "output")

    input_list = ["field", "time", "channel", "plane"]
    all_combinations = []
    for r in range(1, len(input_list) + 1):
        combinations_r = itertools.combinations(input_list, r)
        all_combinations.extend(combinations_r)

    for root_dir in root_dir_list:
        for combination in all_combinations:
            convert_md_to_ome_tiffs(
                input_dir=root_dir,
                output_dir=temp_dir / "output",
                only_2D=False,
                dimensions_to_split=combination,
                fill_mixed_acquisition=False,
            )
        for combination in all_combinations:
            convert_md_to_ome_tiffs(
                input_dir=root_dir,
                output_dir=temp_dir / "output",
                only_2D=True,
                dimensions_to_split=combination,
                fill_mixed_acquisition=False,
            )
        for combination in all_combinations:
            convert_md_to_ome_tiffs(
                input_dir=root_dir,
                output_dir=temp_dir / "output",
                only_2D=False,
                dimensions_to_split=combination,
                fill_mixed_acquisition=True,
            )
        for combination in all_combinations:
            convert_md_to_ome_tiffs(
                input_dir=root_dir,
                output_dir=temp_dir / "output",
                only_2D=True,
                dimensions_to_split=combination,
                fill_mixed_acquisition=True,
            )
