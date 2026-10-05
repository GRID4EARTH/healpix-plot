from importlib.metadata import version

try:
    __version__ = version("healpix_plot")
except Exception:  # pragma: no cover
    __version__ = "9999"

from healpix_plot.ellipsoid import EllipsoidLike
from healpix_plot.healpix import HealpixGrid
from healpix_plot.plotting import HEALPix, plot
from healpix_plot.resampling import resample
from healpix_plot.sampling_grid import AffineSamplingGrid, SamplingGrid

__all__ = [
    "HealpixGrid",
    "HEALPix",
    "plot",
    "resample",
    "SamplingGrid",
    "AffineSamplingGrid",
    "EllipsoidLike",
]
