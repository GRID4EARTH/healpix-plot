from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from affine import Affine

from healpix_plot.projection import apply_transform

if TYPE_CHECKING:
    from typing import Literal, Self

    from healpix_plot.healpix import HealpixGrid


class SamplingGridParameters(TypedDict):
    """Sampling parameters as a dict

    Parameters
    ----------
    shape : int or tuple of int, default: 1024
        The shape of the array. If a int, the shape is a square of equal size.
    resolution : float or tuple of float, optional
        The resolution or step size of the sampling grid. If a float, expands to
        a 2-tuple of equal values. If missing, derived from the spatial extent
        of the data and the given shape.
    center : tuple of float, optional
        The center of the sampling grid. If missing, this is inferred from the data.
    """

    shape: int | tuple[int, int] = 1024
    resolution: float | tuple[float, float] | None = None
    center: tuple[float, float] | None = None


class SamplingGrid:
    def resolve(
        self, cell_ids: np.ndarray, parameters: HealpixGrid
    ) -> ConcreteSamplingGrid:  # pragma: no cover
        raise NotImplementedError


def canonical_longitude(lon: np.ndarray) -> np.ndarray:
    """Wrap longitudes to the half-open interval ``(-180, 180]``.

    Both edges of the antimeridian are the same meridian; projections map
    ``-180`` and ``180`` to opposite sides of the map, so a single
    representation is needed to decide which side a point belongs to.
    """
    lon = (np.asarray(lon, dtype="float64") + 180.0) % 360.0 - 180.0
    return np.where(lon <= -180.0, lon + 360.0, lon)


def crosses_prime_meridian(cell_ids, params):
    # very coarse detection of prime meridian crossing
    base_cells = cell_ids // 4**params.level
    crossing_base_cells = np.array([0, 3, 4, 8, 11], dtype="uint64")
    return np.any(np.isin(crossing_base_cells, base_cells))


def _infer_parameters(
    grid: ParametrizedSamplingGrid, cell_ids: np.ndarray, params: HealpixGrid
) -> (tuple[int, int], tuple[float, float], tuple[float, float]):
    # TODO: figure out how to deal with the difference between source and target grid
    center = grid.center
    shape = grid.shape
    resolution = grid.resolution
    if resolution is None or center is None:
        lon, lat = params.operations.healpix_to_lonlat(
            cell_ids, **params.as_keyword_params()
        )
        if crosses_prime_meridian(cell_ids, params):
            lon = (lon + 180) % 360 - 180

        if center is None:
            center = (np.mean(lon).item(), np.mean(lat).item())

        if resolution is None:
            size_x, size_y = shape
            min_x, max_x = np.min(lon).item(), np.max(lon).item()
            min_y, max_y = np.min(lat).item(), np.max(lat).item()

            dx = (max_x - min_x) / (size_x - 1)
            dy = (max_y - min_y) / (size_y - 1)

            resolution = (dx, dy)

    return shape, resolution, center


@dataclass
class ParametrizedSamplingGrid:
    shape: tuple[int, int]
    resolution: tuple[float, float] | None
    center: tuple[float, float] | None

    @classmethod
    def from_parameters(
        cls,
        shape: int | tuple[int, int],
        resolution: float | tuple[float, float] | None = None,
        center: tuple[float, float] | None = None,
    ) -> Self:
        if isinstance(shape, int):
            shape = (shape, shape)
        if isinstance(resolution, float):
            resolution = (resolution, resolution)

        return cls(shape=shape, resolution=resolution, center=center)

    @classmethod
    def from_dict(cls, mapping: SamplingGridParameters) -> Self:
        return cls.from_parameters(**mapping)

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        shape: int | tuple[int, int],
    ) -> Self:
        if isinstance(shape, int):
            shape = (shape, shape)

        xmin, ymin, xmax, ymax = bbox

        center = (float(np.mean([xmin, xmax])), float(np.mean([ymin, ymax])))

        resolution = (
            (xmax - xmin) / (shape[0] - 1),
            (ymax - ymin) / (shape[1] - 1),
        )

        return cls(shape=shape, center=center, resolution=resolution)

    def resolve(
        self, cell_ids: np.ndarray, parameters: HealpixGrid
    ) -> ConcreteSamplingGrid:
        shape, resolution, center = _infer_parameters(self, cell_ids, parameters)

        size_x, size_y = shape
        resolution_x, resolution_y = resolution
        half_x = size_x // 2
        half_y = size_y // 2
        center_x, center_y = center

        xmin = center_x - half_x * resolution_x
        xmax = center_x + half_x * resolution_x
        ymin = np.clip(center_y - half_y * resolution_y, -90, 90).item()
        ymax = np.clip(center_y + half_y * resolution_y, -90, 90).item()

        if xmin > xmax:
            # prime meridian crossing
            xmin = (xmin + 180) % 360 - 180
            xmax = (xmax + 180) % 360 - 180

        xs = np.linspace(xmin, xmax, size_x, endpoint=True)
        ys = np.linspace(ymin, ymax, size_y, endpoint=True)

        x, y = np.meshgrid(xs, ys)

        extent_x = (xmin, xmax)
        extent_y = (ymin, ymax)

        return ConcreteSamplingGrid(x, y, extent_x, extent_y)


@dataclass
class AffineSamplingGrid(SamplingGrid):
    """A sampling grid defined like a raster: an affine transform and a shape.

    Parameters
    ----------
    transform : affine.Affine
        Maps pixel indices ``(col, row)`` to coordinates, following the GDAL /
        rasterio convention: ``transform * (0, 0)`` is the outer corner of the
        first pixel, and the centre of pixel ``(row, col)`` is at
        ``transform * (col + 0.5, row + 0.5)``.
    shape : tuple of int
        ``(height, width)``, i.e. ``(rows, cols)``, like a raster.
    crs : CRS-like, optional
        The coordinate reference system of ``transform``, as anything accepted
        by :py:meth:`pyproj.CRS.from_user_input` (a :py:class:`pyproj.CRS`, a
        PROJ string, an EPSG code, a rasterio CRS ...). ``None`` means
        geographic coordinates (longitude / latitude in degrees).

    See Also
    --------
    AffineSamplingGrid.from_raster : the grid of an existing raster
    AffineSamplingGrid.from_healpix : the lossless grid in ``+proj=healpix``
    """

    transform: Affine
    shape: tuple[int, int]
    crs: Any = None

    def __post_init__(self):
        if isinstance(self.shape, int):
            self.shape = (self.shape, self.shape)
        self.shape = tuple(int(size) for size in self.shape)

        if self.crs is not None:
            import pyproj

            self.crs = pyproj.CRS.from_user_input(self.crs)

    @classmethod
    def from_transform(
        cls,
        transform: Affine,
        shape: int | tuple[int, int],
        crs: Any = None,
    ) -> Self:
        return cls(transform, shape, crs)

    @classmethod
    def from_raster(cls, raster) -> Self:
        """The sampling grid of an existing raster.

        Parameters
        ----------
        raster : rasterio dataset or xarray object with the ``rio`` accessor
            Anything with ``transform``, ``height``, ``width`` and ``crs``
            attributes (an open :py:class:`rasterio.io.DatasetReader`), or an
            :py:class:`xarray.DataArray` / :py:class:`xarray.Dataset` with the
            ``rioxarray`` accessor. Without a CRS, the ``healpix_crs_wkt`` tag
            written by :py:func:`healpix_plot.raster.to_dataarray` is used.
        """
        # GDAL cannot encode the HEALPix projection in GeoTIFF GeoKeys and keeps
        # it in a side-car; fall back to the WKT tag written by `to_dataarray`
        rio = getattr(raster, "rio", None)
        if rio is not None:
            crs = rio.crs or raster.attrs.get("healpix_crs_wkt")
            return cls(rio.transform(), (rio.height, rio.width), crs)
        crs = raster.crs or raster.tags().get("healpix_crs_wkt")
        return cls(raster.transform, (raster.height, raster.width), crs)

    @classmethod
    def from_healpix(
        cls,
        healpix_grid: HealpixGrid,
        cell_ids: np.ndarray | None = None,
        *,
        lon_0: float = 0.0,
    ) -> Self:
        """The grid in ``+proj=healpix`` with exactly one pixel per cell.

        See :py:func:`healpix_plot.projection.cell_aligned_grid`.
        """
        from healpix_plot.projection import cell_aligned_grid

        return cell_aligned_grid(healpix_grid, cell_ids, lon_0=lon_0)

    @property
    def corner_transform(self) -> Affine:
        return self.transform

    @property
    def center_transform(self) -> Affine:
        return self.transform @ Affine.translation(0.5, 0.5)

    @property
    def is_geographic(self) -> bool:
        return self.crs is None or self.crs.is_geographic

    def pixel_indices(
        self, cell_ids: np.ndarray, healpix_grid: HealpixGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        """The ``(rows, cols)`` of the pixels containing the cell centres.

        Indices may lie outside the grid; compare against ``shape`` to find
        the cells that fall onto it.
        """
        lon, lat = healpix_grid.operations.healpix_to_lonlat(
            np.asarray(cell_ids, dtype="uint64"), **healpix_grid.as_keyword_params()
        )
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        if self.is_geographic:
            x, y = lon, lat
        else:
            import pyproj

            forward = pyproj.Transformer.from_crs(
                self.crs.geodetic_crs, self.crs, always_xy=True
            )
            x, y = forward.transform(canonical_longitude(lon), lat)

        cols, rows = apply_transform(~self.transform, x, y)
        return np.floor(rows).astype("int64"), np.floor(cols).astype("int64")

    def resolve(
        self, cell_ids: np.ndarray, parameters: HealpixGrid
    ) -> ConcreteSamplingGrid:
        height, width = self.shape
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x, y = apply_transform(self.center_transform, cols, rows)

        corner_x, corner_y = apply_transform(
            self.corner_transform,
            np.array([0, width, 0, width]),
            np.array([0, 0, height, height]),
        )
        extent_x = (float(corner_x.min()), float(corner_x.max()))
        extent_y = (float(corner_y.min()), float(corner_y.max()))

        return ConcreteSamplingGrid(
            x, y, extent_x, extent_y, crs=self.crs, transform=self.transform
        )


@dataclass
class ConcreteSamplingGrid:
    """A resolved sampling grid: the coordinates of every pixel centre.

    Parameters
    ----------
    x, y : numpy.ndarray
        2D arrays (``rows``, ``cols``) of pixel centre coordinates in the units
        of ``crs`` (degrees of longitude / latitude if ``crs`` is ``None``).
    extent_x, extent_y : tuple of float
        The outer edges (or bounding box) of the grid along each axis.
    crs : pyproj.CRS, optional
        The coordinate reference system of ``x`` and ``y``. ``None`` means
        geographic coordinates.
    transform : affine.Affine, optional
        The (pixel corner) affine transform, if the grid was defined by one.
    """

    x: np.ndarray
    y: np.ndarray

    extent_x: tuple[float, float]
    extent_y: tuple[float, float]

    crs: Any = None
    transform: Affine | None = None

    @property
    def shape(self):
        return self.x.shape

    @property
    def is_geographic(self) -> bool:
        return self.crs is None or self.crs.is_geographic

    @property
    def is_axis_aligned(self) -> bool:
        """whether the pixel edges are parallel to the coordinate axes"""
        return self.transform is None or (
            self.transform.b == 0 and self.transform.d == 0
        )

    @property
    def origin(self) -> Literal["lower", "upper"]:
        """where row 0 sits, as the ``origin`` argument of ``imshow``"""
        if self.transform is None or self.transform.e > 0:
            return "lower"
        return "upper"

    @property
    def extent(self):
        """``(xmin, xmax, ymin, ymax)``, as the ``extent`` argument of ``imshow``"""
        if self.transform is None:
            extent_x = tuple((x + 180) % 360 - 180 for x in self.extent_x)
        else:
            extent_x = tuple(self.extent_x)
        return extent_x + tuple(self.extent_y)

    @cached_property
    def _lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        if self.crs is None:
            lon, lat = self.x, self.y
            if self.transform is not None:
                lon = (lon + 180) % 360 - 180
            return lon, lat

        import pyproj

        inverse = pyproj.Transformer.from_crs(
            self.crs, self.crs.geodetic_crs, always_xy=True
        )
        lon, lat = inverse.transform(self.x, self.y, errcheck=False)
        return np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")

    @property
    def lon(self) -> np.ndarray:
        """the longitude of the pixel centres (``inf`` off the globe)"""
        return self._lonlat[0]

    @property
    def lat(self) -> np.ndarray:
        """the latitude of the pixel centres (``inf`` off the globe)"""
        return self._lonlat[1]

    @property
    def _pixel_scale(self) -> float:
        """a typical pixel size, in the units of ``x`` and ``y``"""
        if self.transform is not None:
            a, b, _, d, e, *_ = self.transform
            return float(max(abs(a), abs(b), abs(d), abs(e)))
        height, width = self.shape
        dx = float(np.abs(np.diff(self.x, axis=1)).mean()) if width > 1 else 0.0
        dy = float(np.abs(np.diff(self.y, axis=0)).mean()) if height > 1 else 0.0
        return max(dx, dy) or 1.0

    @cached_property
    def valid(self) -> np.ndarray:
        """mask of the pixels whose centre is a (unique) point on the globe

        Pixels of a projected grid can lie outside the projection's domain
        (the inverse projection then returns ``inf``), and the two edges of a
        seam (for instance the antimeridian of a cylindrical projection) map
        to the same point on the globe. Only the pixels whose centre projects
        back onto itself (with longitudes in ``(-180, 180]``) are valid, so
        every point on the globe corresponds to at most one pixel.
        """
        lon, lat = self._lonlat
        valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90)
        if self.is_geographic or not valid.any():
            return valid

        import pyproj

        forward = pyproj.Transformer.from_crs(
            self.crs.geodetic_crs, self.crs, always_xy=True
        )
        x, y = forward.transform(
            canonical_longitude(lon[valid]), lat[valid], errcheck=False
        )
        tolerance = 1e-6 * self._pixel_scale
        roundtrip = (np.abs(np.asarray(x) - self.x[valid]) <= tolerance) & (
            np.abs(np.asarray(y) - self.y[valid]) <= tolerance
        )
        valid[valid] = roundtrip
        return valid

    def cell_ids(self, healpix_grid: HealpixGrid) -> np.ma.MaskedArray:
        """The id of the cell containing each pixel centre.

        Returns a masked array of ``shape``; pixels off the globe are masked.
        """
        lon, lat = self._lonlat
        valid = self.valid

        ids = np.zeros(self.shape, dtype="uint64")
        if valid.any():
            ids[valid] = healpix_grid.operations.lonlat_to_healpix(
                np.ascontiguousarray(lon[valid]),
                np.ascontiguousarray(lat[valid]),
                **healpix_grid.as_keyword_params(),
            )

        return np.ma.MaskedArray(ids, mask=~valid)

    def corners(self) -> tuple[np.ndarray, np.ndarray]:
        """The coordinates of the pixel corners, as ``(rows + 1, cols + 1)`` arrays.

        Suitable for ``pcolormesh(..., shading="flat")``.
        """
        height, width = self.shape
        if self.transform is not None:
            rows, cols = np.meshgrid(
                np.arange(height + 1), np.arange(width + 1), indexing="ij"
            )
            return apply_transform(self.transform, cols, rows)

        xs = self.x[0, :]
        ys = self.y[:, 0]

        def edges(centers):
            inner = (centers[1:] + centers[:-1]) / 2
            first = centers[0] - (inner[0] - centers[0])
            last = centers[-1] + (centers[-1] - inner[-1])
            return np.concatenate([[first], inner, [last]])

        return np.meshgrid(edges(xs), edges(ys))
