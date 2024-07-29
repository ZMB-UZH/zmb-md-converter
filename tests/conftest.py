import subprocess
import zipfile

import pytest


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    # Create a temporary directory that lasts for the session
    base_temp = tmp_path_factory.mktemp("data")

    # Download files from Zenodo into the temporary directory
    subprocess.run(
        [
            "zenodo_get",
            "10.5281/zenodo.12724927",
            "-o",
            str(base_temp),
            "-g",
            "direct_transfer.zip",
        ],
        check=True,
    )
    subprocess.run(
        [
            "zenodo_get",
            "10.5281/zenodo.12724927",
            "-o",
            str(base_temp),
            "-g",
            "MetaXpress_all-z_include-projection.zip",
        ],
        check=True,
    )

    # Unzip the downloaded files
    for zip_file in ["direct_transfer.zip", "MetaXpress_all-z_include-projection.zip"]:
        zip_path = base_temp / zip_file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(base_temp)

    return base_temp
