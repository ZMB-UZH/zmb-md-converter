"""A library and small GUI to convert data from the MD-ImageXpress microscope."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("zmb-md-converter")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Flurin Sturzenegger"
__email__ = "st.flurin@gmail.com"
