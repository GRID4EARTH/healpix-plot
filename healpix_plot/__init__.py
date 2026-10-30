from importlib.metadata import version

try:
    __version__ = version("healpix_plot")
except Exception:  # pragma: no cover
    __version__ = "9999"

from healpix_plot.plotting import contour, contourf, imshow, plot
from healpix_plot.resampling import resample
from healpix_plot.sampling_grid import SamplingGrid

__all__ = [
    "SamplingGrid",
    "contour",
    "contourf",
    "imshow",
    "plot",
    "resample",
]
