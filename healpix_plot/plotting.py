from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from healpix_geo.auto import Grid
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Norm
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from healpix_plot.resampling import AggType, resample
from healpix_plot.sampling_grid import (
    BoxSamplingGrid,
    BoxSamplingGridDict,
    SamplingGrid,
)


def plot(
    cell_ids: NDArray[np.integer],
    data: np.ndarray,
    *,
    healpix_grid: Grid | Mapping,
    sampling_grid: SamplingGrid | BoxSamplingGridDict,
    projection: str | ccrs.CRS = "Mollweide",
    view: tuple[float, float, float, float] | None = None,
    agg: AggType = "mean",
    interpolation: Literal["nearest", "bilinear"] = "nearest",
    background_value: float = np.nan,
    ax: GeoAxes | None = None,
    title: str | None = None,
    colorbar: bool | dict[str, Any] = False,
    cmap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Norm | None = None,
    axis_labels: dict[str, str] | Literal["none"] | None = None,
) -> GeoAxes:
    """Resample and plot HEALPix data.

    Parameters
    ----------
    cell_ids : array-like of int
        The HEALPix cell indices, in the indexing scheme specified by `healpix_grid`.
    data : array-like of float
        The data to plot. Supported shapes are:

        - (N): Scalar data mapped to colors
        - (N, 3): RGB values (0-1 float)
        - (N, 4): RGBA values (0-1 float)
    healpix_grid : Grid or dict of str to any
        The healpix grid parameters necessary to interpret ``cell_ids``. A grid object
        or a mapping of parameters `level`, `indexing_scheme` and `ellipsoid`.
    sampling_grid : SamplingGrid or BoxSamplingGridDict
        The target grid. Can be specified as a subclass of :class:`SamplingGrid` or
        a mapping of parameters for :class:`BoxSamplingGrid`.
    projection : str or cartopy.crs.CRS
        The projection used to construct a new axis. Ignored if ``ax`` is given.
    view : tuple of float, optional
        If given, defines the extent of the displayed plot as (xmin, ymin, xmax, ymax).
    agg : str, default: "mean"
        Aggregation method to apply to HEALPix cells that fall into the same target
        cell. Will be passed to `numpy_groupies.reduce`. Can be any of "mean", "median",
        "std", "var", "min", "max", "first", "last".
    interpolation: {"nearest", "bilinear"}, default: "nearest"
        The algorithm used to interpolate from healpix to the target grid. Available
        values:

        - ``"nearest"``: nearest-neighbour resampling
        - ``"bilinear"``: bilinear resampling
    background_value : float, default: numpy.nan
        The background value for missing values.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If not passed, a new figure with a single axis is
        created using ``projection`` and ``figure_params``.
    cmap: Colormap or str, default: "viridis"
        The Colormap instance or registered colormap name used to map scalar data to
        colors. This parameter is ignored if the data is RGB(A).
    vmin, vmax : float, optional
        Data range that the colormap should cover. See the
        :meth:`~matplotlib.axes.Axes.imshow` documentation for details.
    norm: matplotlib.colors.Norm or str, optional
       The normalization method used to scale scalar data to the [0, 1] range before
       mapping to colors using cmap. See the :meth:`~matplotlib.axes.Axes.imshow`
       documentation for details.
    title: str, optional
        Title to add to the figure.
    colorbar: bool or dict of str to any, default: False
        If True, add a colorbar to the figure. If a dict, pass those parameters to
        :meth:`Figure.colorbar <matplotlib.figure.Figure.colorbar>`.
    axis_labels : dict of str to str or "none", optional
        Labels to add to the axes. Possible values:

        - if ``None`` or not passed, ``"Longitude"`` and ``"Latitude"`` are used.
        - dict: the keys ``"x"`` and ``"y"`` are used
        - ``"none"``: no axis labels

    Returns
    -------
    mappable : matplotlib.image.AxesImage
        The mappable of the image to allow further processing.

    Examples
    --------
    >>> import healpix_plot
    >>> import numpy as np

    Define the source grid:

    >>> healpix_grid = healpix_geo.auto.Grid(
    ...     level=4,
    ...     indexing_scheme="nested",
    ... )
    >>> cell_ids = np.arange(12 * 4**healpix_grid.level, dtype="uint64")

    Create the data:

    >>> lon, lat = healpix_geo.auto.healpix_to_lonlat(cell_ids, healpix_grid)
    >>> data = np.cos(8 * np.deg2rad(lon)) * np.sin(4 * np.deg2rad(lat))

    Plot the data

    >>> healpix_plot.plot(
    ...     cell_ids,
    ...     data,
    ...     healpix_grid=healpix_grid,
    ...     sampling_grid={"shape": 1024},
    ... )  # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage at 0x...>
    """
    if not isinstance(sampling_grid, SamplingGrid):
        sampling_grid = BoxSamplingGrid(**sampling_grid)
    if not isinstance(healpix_grid, Grid):
        healpix_grid = Grid(**healpix_grid)

    target_grid, image = resample(
        cell_ids,
        data,
        sampling_grid=sampling_grid,
        healpix_grid=healpix_grid,
        interpolation=interpolation,
        agg=agg,
        background_value=background_value,
    )

    if isinstance(projection, str):
        _projection = getattr(ccrs, projection, None)
        if _projection is None:
            raise ValueError(f"unknown projection: {projection}")
        projection = _projection()

    if ax is None:
        _, ax = plt.subplots(
            figsize=(12, 10),
            subplot_kw={"projection": projection},
            layout="constrained",
        )
        assert isinstance(ax, GeoAxes)

    if healpix_grid.level is not None and cell_ids.size == 12 * 4**healpix_grid.level:
        ax.set_global()
    elif view is not None:
        ax.set_extent(view, crs=ccrs.PlateCarree())
    else:
        # set extent before plotting for a smoother image
        # See https://github.com/SciTools/cartopy/issues/1468
        ax.set_extent(target_grid.extent, crs=ccrs.PlateCarree())

    mappable = ax.imshow(
        image,
        extent=target_grid.extent,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
    )
    if title is not None:
        ax.set_title(title)

    if colorbar:
        colorbar_kwargs = colorbar if isinstance(colorbar, dict) else {}
        ax.figure.colorbar(mappable, **colorbar_kwargs)

    if axis_labels != "none":
        if axis_labels is None:
            axis_labels = {"x": "Longitude", "y": "Latitude"}

        for axis in ["x", "y"]:
            getattr(ax, f"set_{axis}label")(axis_labels[axis])

    return mappable


def imshow(
    cell_ids: NDArray[np.integer],
    data: np.ndarray,
    *,
    healpix_grid: Grid | Mapping,
    sampling_grid: SamplingGrid | BoxSamplingGridDict,
    interpolation: Literal["nearest", "bilinear"] = "nearest",
    agg: AggType = "mean",
    background_value: float = np.nan,
    ax: Axes | None = None,
    **kwargs: Any,
) -> AxesImage:
    """Resample HEALPix data and plot it using imshow.

    Parameters
    ----------
    cell_ids : array-like of int
        The HEALPix cell indices, in the indexing scheme specified by `healpix_grid`.
    data : array-like of float
        The data to plot. Supported shapes are:

        - (N): Scalar data mapped to colors
        - (N, 3): RGB values (0-1 float)
        - (N, 4): RGBA values (0-1 float)
    sampling_grid : SamplingGrid or mapping
        The target grid.
    agg : {"mean", "median", "std", "var", "min", "max", "first", "last"}, default: "mean"
        Aggregation method to deduplicate the data.
    background_value: float, default: numpy.nan
        The background value for missing values.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot unto. If left to None, the current axes is used.
    kwargs :
        Passed to :meth:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    mappable: matplotlib.image.AxesImage
    """
    target_grid, image = resample(
        cell_ids,
        data,
        healpix_grid=healpix_grid,
        sampling_grid=sampling_grid,
        interpolation=interpolation,
        agg=agg,
        background_value=background_value,
    )

    if ax is None:
        ax = plt.gca()

    kwargs: dict[str, Any] = {"interpolation": "nearest", "aspect": "auto"} | kwargs

    return ax.imshow(image, extent=target_grid.extent, origin="lower", **kwargs)


def contour(
    cell_ids: NDArray[np.integer],
    data: np.ndarray,
    *,
    healpix_grid: Grid | Mapping,
    sampling_grid: SamplingGrid | BoxSamplingGridDict,
    interpolation: Literal["nearest", "bilinear"] = "nearest",
    agg: AggType = "mean",
    background_value: float = np.nan,
    ax: Axes | None = None,
    **kwargs: Any,
) -> QuadContourSet:
    """Resample HEALPix data and plot contour lines.

    Parameters
    ----------
    cell_ids : array-like of int
        The HEALPix cell indices, in the indexing scheme specified by `healpix_grid`.
    data : array-like of float
        The data to plot.
    sampling_grid : SamplingGrid or mapping
        The target grid.
    agg : {"mean", "median", "std", "var", "min", "max", "first", "last"}, default: "mean"
        Aggregation method to deduplicate the data.
    background_value: float, default: numpy.nan
        The background value for missing values.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot unto. If left to None, the current axes is used.
    kwargs :
        Passed to :meth:`matplotlib.axes.Axes.contour`.

    Returns
    -------
    mappable: matplotlib.contour.QuadContourSet
    """
    target_grid, image = resample(
        cell_ids,
        data,
        healpix_grid=healpix_grid,
        sampling_grid=sampling_grid,
        interpolation=interpolation,
        agg=agg,
        background_value=background_value,
    )

    if ax is None:
        ax = plt.gca()

    return ax.contour(target_grid.x, target_grid.y, image, **kwargs)


def contourf(
    cell_ids: np.ndarray,
    data: np.ndarray,
    *,
    healpix_grid: Grid | Mapping,
    sampling_grid: SamplingGrid | BoxSamplingGridDict,
    interpolation: Literal["nearest", "bilinear"] = "nearest",
    agg: AggType = "mean",
    background_value: float = np.nan,
    ax: Axes | None = None,
    **kwargs: Any,
) -> QuadContourSet:
    """Resample HEALPix data and plot filled contour lines.

    Parameters
    ----------
    cell_ids : array-like of int
        The HEALPix cell indices, in the indexing scheme specified by `healpix_grid`.
    data : array-like of float
        The data to plot.
    sampling_grid : SamplingGrid or mapping
        The target grid.
    agg : {"mean", "median", "std", "var", "min", "max", "first", "last"}, default: "mean"
        Aggregation method to deduplicate the data.
    background_value: float, default: numpy.nan
        The background value for missing values.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot unto. If left to None, the current axes is used.
    kwargs :
        Passed to :meth:`matplotlib.axes.Axes.contourf`.

    Returns
    -------
    mappable: matplotlib.contour.QuadContourSet
    """
    target_grid, image = resample(
        cell_ids,
        data,
        healpix_grid=healpix_grid,
        sampling_grid=sampling_grid,
        interpolation=interpolation,
        agg=agg,
        background_value=background_value,
    )

    if ax is None:
        ax = plt.gca()

    return ax.contourf(target_grid.x, target_grid.y, image, **kwargs)
