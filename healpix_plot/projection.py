"""PROJ helpers for the ellipsoidal HEALPix projection (``+proj=healpix``).

PROJ's ``healpix`` projection composes the authalic-latitude conversion with
the spherical HPX projection, which is exactly how ``healpix-geo`` defines
ellipsoidal HEALPix cells. In that plane every cell of a given level is a
square rotated by 45°, so a raster with one pixel per cell can be described by
an affine transform, and written to a GeoTIFF without loss.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from affine import Affine

if TYPE_CHECKING:
    import pyproj

    from healpix_plot.healpix import HealpixGrid
    from healpix_plot.sampling_grid import AffineSamplingGrid


def resolve_ellipsoid(ellipsoid: Any) -> dict[str, Any]:
    """Normalize an ellipsoid-like object to a healpix-geo style dict.

    Accepts a name known to ``healpix_geo.ellipsoid.resolve`` (``"sphere"``,
    ``"WGS84"``, ...), a dict with ``radius`` or ``semimajor_axis`` /
    ``inverse_flattening``, an object with those attributes, an object with an
    ``ellipsoid`` attribute (``HealpixGrid`` or ``xdggs.HealpixInfo``), or
    ``None`` (the sphere).
    """
    import healpix_geo.ellipsoid

    if ellipsoid is None:
        ellipsoid = "sphere"
    if hasattr(ellipsoid, "ellipsoid") and not isinstance(ellipsoid, dict):
        ellipsoid = ellipsoid.ellipsoid
        if ellipsoid is None:
            ellipsoid = "sphere"
    if isinstance(ellipsoid, str):
        return dict(healpix_geo.ellipsoid.resolve(ellipsoid))
    if isinstance(ellipsoid, dict):
        return dict(ellipsoid)
    if hasattr(ellipsoid, "radius"):
        return {"radius": float(ellipsoid.radius)}
    return {
        "semimajor_axis": float(ellipsoid.semimajor_axis),
        "inverse_flattening": float(ellipsoid.inverse_flattening),
    }


def _shape_params(ellipsoid: Any) -> str:
    params = resolve_ellipsoid(ellipsoid)
    if "radius" in params:
        return f"+R={params['radius']!r}"
    if "semiminor_axis" in params and "inverse_flattening" not in params:
        return f"+a={params['semimajor_axis']!r} +b={params['semiminor_axis']!r}"
    return f"+a={params['semimajor_axis']!r} +rf={params['inverse_flattening']!r}"


def proj_string(ellipsoid: Any = "sphere", lon_0: float = 0.0) -> str:
    """The PROJ string of the HEALPix projection on the given ellipsoid.

    The ellipsoid parameters are taken from ``healpix-geo``, so PROJ and
    ``healpix-geo`` use exactly the same figure of the Earth.
    """
    return f"+proj=healpix {_shape_params(ellipsoid)} +lon_0={float(lon_0)!r}"


def geographic_proj_string(ellipsoid: Any = "sphere") -> str:
    """The PROJ string of geographic (lon / lat) coordinates on the ellipsoid."""
    return f"+proj=longlat {_shape_params(ellipsoid)}"


def healpix_crs(ellipsoid: Any = "sphere", lon_0: float = 0.0) -> pyproj.CRS:
    """``pyproj.CRS`` of the HEALPix projection on the given ellipsoid."""
    import pyproj

    return pyproj.CRS.from_proj4(proj_string(ellipsoid, lon_0))


def geographic_crs(ellipsoid: Any = "sphere") -> pyproj.CRS:
    """``pyproj.CRS`` of geographic coordinates on the given ellipsoid."""
    import pyproj

    return pyproj.CRS.from_proj4(geographic_proj_string(ellipsoid))


def apply_transform(transform: Affine, cols, rows):
    """Apply an affine transform to arrays of pixel coordinates."""
    a, b, c, d, e, f, *_ = transform
    cols = np.asarray(cols, dtype="float64")
    rows = np.asarray(rows, dtype="float64")
    return a * cols + b * rows + c, d * cols + e * rows + f


def plane_half_width(crs: pyproj.CRS, lon_0: float = 0.0) -> float:
    """Half the width of the HEALPix map (π times the authalic radius)."""
    import pyproj

    forward = pyproj.Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    x, _ = forward.transform(lon_0 + 90.0, 0.0)
    return 2.0 * float(x)


def cell_aligned_grid(
    healpix_grid: HealpixGrid,
    cell_ids: np.ndarray | None = None,
    *,
    lon_0: float = 0.0,
) -> AffineSamplingGrid:
    """A sampling grid in ``+proj=healpix`` with exactly one pixel per cell.

    The pixels are squares rotated by 45° that coincide with the HEALPix cells
    of ``healpix_grid.level``, so resampling onto this grid is lossless. The
    full grid covers the bounding box of the 12 base facets in the rotated
    frame (``6 * nside`` pixels on each side, a third of which lie on the
    globe); passing ``cell_ids`` crops it to the bounding box of those cells.

    ``lon_0`` must be a multiple of 90° for the cells to line up with the
    pixels, and ``level`` must be at least 1: in the rotated frame the polar
    facets are offset from the equatorial ones by half a facet, so the base
    cells themselves do not share a common pixel lattice.
    """
    from healpix_plot.sampling_grid import AffineSamplingGrid

    if healpix_grid.level is None or healpix_grid.indexing_scheme == "zuniq":
        raise ValueError("a cell-aligned grid needs a single resolution level")
    if healpix_grid.level < 1:
        raise ValueError(
            "the base cells (level 0) do not align with a common pixel grid;"
            " use level 1 or higher"
        )
    if lon_0 % 90 != 0:
        raise ValueError(
            f"lon_0 must be a multiple of 90° for the cells to align, got {lon_0}"
        )

    crs = healpix_crs(healpix_grid.ellipsoid, lon_0)
    half_width = plane_half_width(crs, lon_0)
    nside = 2**healpix_grid.level
    # base facets are squares with a diagonal of a quarter of the map width
    facet_half_diagonal = half_width / 4.0
    cell_side = np.sqrt(2.0) * facet_half_diagonal / nside
    k = cell_side / np.sqrt(2.0)
    # columns run towards south-east, rows towards south-west; pixel (0, 0) is
    # the north-eastern corner of the bounding box of the staircase of facets
    transform = Affine(k, -k, 0.0, -k, -k, 1.5 * half_width)
    shape = (6 * nside, 6 * nside)

    grid = AffineSamplingGrid(transform, shape, crs)
    if cell_ids is not None:
        rows, cols = grid.pixel_indices(cell_ids, healpix_grid)
        row0, col0 = int(rows.min()), int(cols.min())
        shape = (int(rows.max()) - row0 + 1, int(cols.max()) - col0 + 1)
        grid = AffineSamplingGrid(
            transform @ Affine.translation(col0, row0), shape, crs
        )

    return grid
