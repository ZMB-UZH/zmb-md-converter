"""Unit tests for standalone helper functions in convert.py."""

from zmb_md_converter.convert import _sanitize_output_name


def test_sanitize_output_name():
    assert _sanitize_output_name("my_plate 2026-08-25, 14:01:21") == (
        "my_plate_2026-08-25_14_01_21"
    )
    assert _sanitize_output_name("A02") == "A02"
