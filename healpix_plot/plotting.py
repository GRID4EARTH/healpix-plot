from __future__ import annotations

from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeom

from healpix_plot.healpix import HealpixGrid
from healpix_plot.projection import resolve_ellipsoid
from healpix_plot.resampling import is_rgb, resample
from healpix_plot.sampling_grid import ParametrizedSamplingGrid

if TYPE_CHECKING:
    from typing import Any, Literal

    import cartopy.crs as ccrs
    from matplotlib.axis import Axis
    from matplotlib.cm import ColorMap
    from matplotlib.norm import Norm

    from healpix_plot.sampling_grid import SamplingGrid, SamplingGridParameters


class HEALPix(ccrs.Projection):
    """Cartopy projection for PROJ's ellipsoidal HEALPix (``+proj=healpix``).

    Cartopy needs the outline of the map in addition to the PROJ definition;
    this class supplies it (the equatorial band plus the four polar triangles
    at each pole), so coastlines, features, gridlines and data passed with
    ``transform=ccrs.PlateCarree()`` all work.

    Parameters
    ----------
    central_longitude : float, default: 0.0
        The central meridian (``lon_0``). Only multiples of 90° line up with
        the HEALPix cells.
    ellipsoid : ellipsoid-like, default: "sphere"
        The reference ellipsoid, resolved like ``HealpixGrid.ellipsoid``. Pass
        the same ellipsoid as the data for the cells to be squares.
    """

    def __init__(self, central_longitude: float = 0.0, ellipsoid: Any = "sphere"):
        params = resolve_ellipsoid(ellipsoid)
        if "radius" in params:
            globe = ccrs.Globe(
                ellipse=None,
                semimajor_axis=params["radius"],
                semiminor_axis=params["radius"],
            )
        else:
            globe = ccrs.Globe(
                ellipse=None,
                semimajor_axis=params["semimajor_axis"],
                inverse_flattening=params["inverse_flattening"],
            )
        super().__init__(
            [("proj", "healpix"), ("lon_0", central_longitude)], globe=globe
        )

        # x spans [-πR, πR] (R: authalic radius); x(lon_0 + 90°) = πR / 2
        w = (
            2.0
            * self.transform_point(central_longitude + 90.0, 0.0, self.as_geodetic())[0]
        )
        q = w / 4.0
        self._half_width = w

        # outline, counter-clockwise from the bottom-left corner: the equatorial
        # band |y| <= w/4 plus a triangle of height w/4 over each polar facet
        xs = np.linspace(-w, w, 9)
        top = [(x, q if i % 2 == 0 else 2 * q) for i, x in enumerate(xs)]
        bottom = [(x, -q if i % 2 == 0 else -2 * q) for i, x in enumerate(xs)]
        self._boundary = sgeom.LinearRing(bottom + top[::-1])
        self._threshold = w / 1e4

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return (-self._half_width, self._half_width)

    @property
    def y_limits(self):
        return (-self._half_width / 2, self._half_width / 2)

    @property
    def threshold(self):
        return self._threshold


def _data_crs(target_grid, ax=None) -> ccrs.CRS:
    """the cartopy CRS of the resampled image

    Cartopy only draws images and meshes from a `Projection` (not a bare
    `CRS`). When the axes already use the same projection, that instance is
    returned so that cartopy does not re-project the data at all.
    """
    if target_grid.is_geographic:
        return ccrs.PlateCarree()

    projection = ccrs.Projection(target_grid.crs)
    if ax is not None and getattr(ax, "projection", None) == projection:
        return ax.projection
    return projection


def plot(
    cell_ids: np.ndarray,
    data: np.ndarray,
    *,
    healpix_grid: HealpixGrid,
    sampling_grid: SamplingGridParameters | SamplingGrid,
    projection: str | ccrs.CRS = "Mollweide",
    view: tuple[float, float, float, float] | None = None,
    agg: str = "mean",
    interpolation: str = "nearest",
    background_value: float = np.nan,
    rgb_clip: tuple[float, float] = (0.0, 1.0),
    ax: Axis | None = None,
    title: str | None = None,
    colorbar: bool | dict[str, Any] = False,
    cmap: str | ColorMap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Norm | None = None,
    axis_labels: dict[str, str] | Literal["none"] | None = None,
) -> Axis:
    """resample and plot healpix data

    Parameters
    ----------
    cell_ids : numpy.ndarray
        The cell ids describing the spatial position of the data.
    data : numpy.ndarray
        The data to plot. If 1D, will be color-coded using the standard
        matplotlib mechanisms. If 2D, the last axis must have a size of 3 (for
        RGB) or 4 (for RGBA).
    healpix_grid : HealpixGrid or dict of str to any
        The healpix grid parameters necessary to interpret ``cell_ids``.
    sampling_grid : SamplingGrid or dict of str to any
        The target grid.
    projection : str or cartopy.crs.CRS
        The projection used to construct a new axis. Ignored if ``ax`` is
        given. The name of a cartopy projection, or ``"HEALPix"`` for
        :py:class:`HEALPix` on the data's ellipsoid.
    view : tuple of float, optional
        If given, defines the extent of the displayed plot.
    agg : str, default: "mean"
        Aggregation to deduplicate the data.
    interpolation : str, default: "nearest"
        The algorithm used to interpolate from healpix to the target grid. Available values:

        - ``"nearest"``: nearest-neighbour resampling
        - ``"bilinear"``: bilinear resampling

    background_value : float, default: numpy.nan
        The background value for missing values.
    ax : matplotlib.axis.Axis, optional
        The axis to plot on. If not passed, a new figure with a single axis is
        created using ``projection`` and ``figure_params``.
    vmin : float, optional
        Minimum value to color-code.
    vmax : float, optional
        Maximum value to color-code.
    norm : matplotlib.norm.Norm, optional
        Normalization class for more control.
    cmap : str or matplotlib.colors.Colormap, default: "viridis"
        The colormap to use for plotting.
    axis_labels : dict of str to str or "none", optional
        Axis labels. Possible values:

        - if ``None`` or not passed, ``"Longitude"`` and ``"Latitude"`` are used.
        - dict: the keys ``"x"`` and ``"y"`` are used
        - ``"none"``: no axis labels

    Returns
    -------
    mappable : matplotlib.image.AxisImage
        The mappable of the image to allow further processing.

    Examples
    --------
    >>> import healpix_plot
    >>> import numpy as np

    Define the source grid:

    >>> healpix_params = healpix_plot.HealpixParameters(
    ...     level=4,
    ...     indexing_scheme="nested",
    ... )
    >>> cell_ids = np.arange(12 * 4 ** healpix_params["level"], dtype="uint64")

    Create the data:

    >>> lon, lat = healpix_params.operations.healpix_to_lonlat(
    ...     cell_ids,
    ...     **healpix_params.as_keyword_params(),
    ... )
    >>> data = np.cos(8 * np.deg2rad(lon)) * np.sin(4 * np.deg2rad(lat))

    Plot the data

    >>> healpix_plot.plot(
    ...     cell_ids,
    ...     data,
    ...     sampling_grid={"shape": 1024},
    ...     healpix_grid=healpix_params,
    ... )  # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage at 0x...>
    """
    if isinstance(sampling_grid, dict):
        sampling_grid = ParametrizedSamplingGrid.from_dict(sampling_grid)
    if isinstance(healpix_grid, dict):
        healpix_grid = HealpixGrid(**healpix_grid)

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
        if projection.lower() == "healpix":
            projection = HEALPix(ellipsoid=healpix_grid.ellipsoid)
        else:
            _projection = getattr(ccrs, projection, None)
            if _projection is None:
                raise ValueError(f"unknown projection: {projection}")
            projection = _projection()

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12, 10),
            subplot_kw={"projection": projection},
            layout="constrained",
        )
    data_crs = _data_crs(target_grid, ax)

    if cell_ids.size == 12 * 4**healpix_grid.level:
        ax.set_global()
    elif view is not None:
        ax.set_extent(view, crs=ccrs.PlateCarree())
    else:
        # set extent before plotting for a smoother image
        # See https://github.com/SciTools/cartopy/issues/1468
        ax.set_extent(target_grid.extent, crs=data_crs)

    if target_grid.is_axis_aligned:
        mappable = ax.imshow(
            image,
            extent=target_grid.extent,
            origin=target_grid.origin,
            interpolation="nearest",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            cmap=cmap,
            transform=data_crs,
        )
    else:
        # rotated pixels (e.g. the cell-aligned grid in `+proj=healpix`)
        # cannot be drawn with `imshow`
        if is_rgb(image):
            raise NotImplementedError(
                "RGB(A) images on rotated sampling grids are not supported"
            )
        corner_x, corner_y = target_grid.corners()
        mappable = ax.pcolormesh(
            corner_x,
            corner_y,
            image,
            shading="flat",
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            cmap=cmap,
            transform=data_crs,
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
