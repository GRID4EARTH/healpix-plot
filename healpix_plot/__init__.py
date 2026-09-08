from importlib.metadata import version

try:
    __version__ = version("healpix_plot")
except Exception:  # pragma: no cover
    __version__ = "9999"

from healpix_plot.ellipsoid import EllipsoidLike
from healpix_plot.healpix import HealpixGrid
from healpix_plot.pcolormesh import pcolormesh
from healpix_plot.plotting import plot
from healpix_plot.resampling import resample
from healpix_plot.sampling_grid import SamplingGrid

__all__ = [
    "EllipsoidLike",
    "HealpixGrid",
    "SamplingGrid",
    "pcolormesh",
    "plot",
    "resample",
]
