import subprocess
import zipfile
from pathlib import Path

import pytest

_TESTS_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def ht_ai_temp_dir():
    base = _TESTS_DATA_DIR / "ht_ai"
    base.mkdir(parents=True, exist_ok=True)

    for filename in [
        "direct_transfer.zip",
        "MetaXpress_all-z_include-projection.zip",
    ]:
        if not (base / filename).exists():
            subprocess.run(
                ["zenodo_get", "10.5281/zenodo.12724927", "-o", str(base), "-g", filename],
                check=True,
            )
        extracted = base / filename.removesuffix(".zip")
        if not extracted.exists():
            with zipfile.ZipFile(base / filename, "r") as zip_ref:
                zip_ref.extractall(base)

    return base


@pytest.fixture(scope="session")
def hcs_ai_temp_dir():
    base = _TESTS_DATA_DIR / "hcs_ai"
    base.mkdir(parents=True, exist_ok=True)

    zip_name = "hcs-ai_minimal.zip"
    if not (base / zip_name).exists():
        subprocess.run(
            ["zenodo_get", "10.5281/zenodo.19694932", "-o", str(base), "-g", zip_name],
            check=True,
        )
    extracted = base / "hcs-ai_minimal"
    if not extracted.exists():
        with zipfile.ZipFile(base / zip_name, "r") as zip_ref:
            zip_ref.extractall(base)

    return extracted
