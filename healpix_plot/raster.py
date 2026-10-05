"""Bridge between resampled images and georeferenced rasters.

These helpers wrap the result of :py:func:`healpix_plot.resample` into an
:py:class:`xarray.DataArray` that ``rioxarray`` understands (and so can write
to a GeoTIFF with ``.rio.to_raster()``), and decode the cell ids of an
existing raster. Neither ``xarray`` nor ``rioxarray`` are hard dependencies;
they are imported on use.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from healpix_plot.projection import geographic_crs, resolve_ellipsoid
from healpix_plot.sampling_grid import AffineSamplingGrid

#: TIFF tag holding the CRS as WKT. GeoTIFF has no GeoKeys for the HEALPix
#: projection, so GDAL keeps the CRS in a ``.aux.xml`` side-car file; the tag
#: survives when the side-car is lost.
CRS_TAG = "healpix_crs_wkt"

if TYPE_CHECKING:
    import xarray as xr

    from healpix_plot.healpix import HealpixGrid
    from healpix_plot.sampling_grid import ConcreteSamplingGrid


def to_dataarray(
    image: np.ndarray,
    target_grid: ConcreteSamplingGrid,
    healpix_grid: HealpixGrid,
    *,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Wrap a resampled image into a georeferenced :py:class:`xarray.DataArray`.

    The result has ``y`` and ``x`` dimensions (plus ``band`` for RGB(A)
    images). With ``rioxarray`` installed, the CRS and the affine transform are
    written with the ``rio`` accessor, so ``result.rio.to_raster(path)``
    produces a GeoTIFF; otherwise they are stored as the ``crs_wkt`` and
    ``transform`` attributes.

    The attributes (written as TIFF tags by rioxarray) also carry the CRS as
    WKT (``healpix_crs_wkt``) and the grid parameters (``healpix_depth``,
    ``healpix_indexing_scheme``, ``healpix_ellipsoid``): GeoTIFF cannot encode
    the HEALPix projection in its GeoKeys, so GDAL stores the CRS in a
    ``<file>.aux.xml`` side-car, and :py:meth:`AffineSamplingGrid.from_raster`
    falls back to the tag when that side-car is missing.

    Parameters
    ----------
    image : numpy.ndarray
        The image returned by :py:func:`healpix_plot.resample`.
    target_grid : ConcreteSamplingGrid
        The resolved sampling grid returned together with the image.
    healpix_grid : HealpixGrid
        The source grid; its ellipsoid defines the geographic CRS of
        unprojected sampling grids.
    """
    import xarray as xr

    dims = ("y", "x") + (("band",) if image.ndim == 3 else ())
    height, width = target_grid.shape[:2]
    if target_grid.is_axis_aligned:
        coords = {"x": target_grid.x[0, :], "y": target_grid.y[:, 0]}
    else:
        # rotated pixels have no 1D coordinates; the transform carries the geolocation
        coords = {"x": np.arange(width), "y": np.arange(height)}

    crs = target_grid.crs
    if crs is None:
        crs = geographic_crs(healpix_grid.ellipsoid)

    ellipsoid = resolve_ellipsoid(healpix_grid.ellipsoid)
    tags = {
        CRS_TAG: crs.to_wkt(),
        "healpix_depth": healpix_grid.level,
        "healpix_indexing_scheme": healpix_grid.indexing_scheme,
        "healpix_ellipsoid": ellipsoid.get("name")
        or json.dumps({k: v for k, v in ellipsoid.items() if k != "name"}),
    }
    arr = xr.DataArray(
        image, dims=dims, coords=coords, name=name, attrs={**tags, **(attrs or {})}
    )

    try:
        import rioxarray  # noqa: F401
    except ImportError:
        arr.attrs["crs_wkt"] = crs.to_wkt()
        if target_grid.transform is not None:
            arr.attrs["transform"] = tuple(target_grid.transform)[:6]
        return arr

    arr = arr.rio.write_crs(crs)
    if target_grid.transform is not None:
        arr = arr.rio.write_transform(target_grid.transform)
    return arr


def raster_cell_ids(raster, healpix_grid: HealpixGrid) -> np.ma.MaskedArray:
    """The cell id under each pixel centre of an existing raster.

    Parameters
    ----------
    raster : rasterio dataset or xarray object with the ``rio`` accessor
        See :py:meth:`AffineSamplingGrid.from_raster`.
    healpix_grid : HealpixGrid
        The grid to express the pixel positions in.

    Returns
    -------
    cell_ids : numpy.ma.MaskedArray
        One cell id per pixel; pixels off the globe are masked. For a raster
        written on :py:meth:`AffineSamplingGrid.from_healpix`, this recovers
        the original cell ids exactly.
    """
    grid = AffineSamplingGrid.from_raster(raster)
    return grid.resolve(None, healpix_grid).cell_ids(healpix_grid)
