from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import healpix_geo
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colorizer import Colorizer
from matplotlib.transforms import Bbox

from healpix_plot.healpix import HealpixGrid

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import Colormap, Norm
    from numpy.typing import NDArray


def wrap_longitude(lon: NDArray, base_cell: int) -> NDArray:
    """Map longitude to [-180, 180]."""
    lon = (lon + 180) % 360 - 180
    if base_cell in [1, 9]:
        lon[lon == -180] = 180
    return lon


def find_extent(cell_ids: NDArray, healpix_grid: HealpixGrid) -> Bbox:
    """Find extent of given cells.

    All cells should be in the same base cell (level 0).
    """
    lon, lat = np.stack(
        healpix_geo.nested.healpix_to_lonlat(cell_ids, **healpix_grid.as_keyword_params()),
        axis=0,
    )
    base_cell = healpix_geo.nested.zoom_to(cell_ids[0], healpix_grid.level, 0)
    # central cell, need to map longitude in [-180, 180]
    if base_cell == 4:
        lon = wrap_longitude(lon, base_cell)

    i_xmin = np.argmin(lon)
    i_xmax = np.argmax(lon)
    i_ymin = np.argmin(lat)
    i_ymax = np.argmax(lat)

    vlon, vlat = healpix_geo.nested.vertices(
        cell_ids[[i_xmin, i_ymin, i_xmax, i_ymax]], **healpix_grid.as_keyword_params()
    )
    if base_cell != 6:
        vlon = wrap_longitude(vlon, base_cell)

    return Bbox(
        [
            [vlon[0, 3], vlat[1, 0]],
            [vlon[2, 1], vlat[3, 2]],
        ]
    )


def _find_tiles(cell_ids: NDArray, level: int, level_tile: int) -> tuple[NDArray, NDArray]:
    """Find tile ids and splits to select them."""
    cell_tile_ids = healpix_geo.nested.zoom_to(cell_ids, level, level_tile)
    tile_ids, tile_splits = np.unique(cell_tile_ids, return_index=True)
    tile_splits = np.append(tile_splits, [None])
    return tile_ids, tile_splits


def _extract_tile(
    cell_ids: NDArray, cell_data: NDArray, splits: NDArray, i_tile: int
) -> tuple[NDArray, NDArray]:
    """Extract tile ids and data."""
    slc = slice(splits[i_tile], splits[i_tile + 1])
    return cell_ids[slc], cell_data[slc]


def _fill_tile(
    tile_id: np.uint64,
    cell_ids_partial: NDArray,
    cell_colors_partial: NDArray,
    level: int,
    level_tile: int,
) -> tuple[NDArray, NDArray]:
    """Fill tile if necessary.

    The parent tile may have missing values (not present in input data). Return RGBA
    array with transparent values for missing cells.
    """
    n_cells = 4 ** (level - level_tile)
    if cell_ids_partial.shape[0] == n_cells:
        return cell_ids_partial, cell_colors_partial

    # We need to fill missing cells in tile. We need the index of each cell,
    # relative to the parent tile
    # Starting id for tile (at fine level)
    tile_id0 = np.left_shift(tile_id, 2 * (level - level_tile))
    ids_relative = cell_ids_partial - tile_id0
    ids = np.arange(n_cells, dtype=cell_ids_partial.dtype) + tile_id0

    color_dtype = cell_colors_partial.dtype
    # If only RGB, add alpha channel
    if cell_colors_partial.shape[1] == 3:
        cell_colors_partial = np.hstack(
            [cell_colors_partial, np.ones([cell_colors_partial.shape[0], 1], dtype=color_dtype)]
        )

    colors = np.ones((n_cells, cell_colors_partial.shape[1]), dtype=color_dtype)
    colors[:, 3] = 0
    colors[ids_relative] = cell_colors_partial

    return ids, colors


def _build_2d_indices(size: int) -> NDArray:
    """Build indices used to reshape tiles to 2d.

    Parameters
    ----------
    size:
        Size of tiles as a healpix level.
    """
    n_side = 2 ** (size)
    _, ix, iy = healpix_geo.nested.pix2xyf(np.arange(4 ** (size)), depth=size)
    return ix + n_side * iy


def _reindex_tile_2d(
    cell_ids: NDArray,
    cell_colors: NDArray,
    n_side: int,
    nested2d_indices: NDArray,
) -> tuple[NDArray, NDArray]:
    """Reindex tile from nested to 2D."""
    n_rgb = cell_colors.shape[1]
    cell_colors_2d = np.zeros((n_side * n_side, n_rgb), dtype=cell_colors.dtype)
    cell_colors_2d[nested2d_indices] = cell_colors
    cell_colors_2d = cell_colors_2d.reshape((n_side, n_side, n_rgb))

    cell_ids_2d = np.zeros(n_side * n_side, dtype="u8")
    cell_ids_2d[nested2d_indices] = cell_ids
    cell_ids_2d = cell_ids_2d.reshape((n_side, n_side))

    return cell_ids_2d, cell_colors_2d


def _compute_grid(
    cell_ids_2d: NDArray, n_side: int, healpix_grid: HealpixGrid
) -> tuple[NDArray, NDArray]:
    """Compute the grid coordinates for pcolormesh.

    Returns
    -------
    lon, lat:
        Coordinates as 2D arrays each of size `(n_side+1, n_side+1)`.
    """
    vertices = np.stack(
        healpix_geo.nested.vertices(cell_ids_2d, **healpix_grid.as_keyword_params()),
        axis=0,
    )

    grid = np.zeros((2, n_side + 1, n_side + 1))

    # Grid "top" is towards north-west, x is fast index.
    # Bottom vertex for bottom-right part of grid
    grid[:, :-1, :-1] = vertices[:, :, :, 0]
    # Right vertex for last column
    grid[:, :-1, -1] = vertices[:, :, -1, 1]
    # Left vertex for top row
    grid[:, -1, :-1] = vertices[:, -1, :, 3]
    # Top vertex for top-right cell
    grid[:, -1, -1] = vertices[:, -1, -1, 2]

    lon, lat = grid[0], grid[1]

    base_cell = healpix_geo.nested.zoom_to(cell_ids_2d[0, 0], healpix_grid.level, 0)
    if base_cell != 6:
        lon = wrap_longitude(lon, base_cell)

    return lon, lat


def _to_sorted(cell_ids: NDArray, cell_data: NDArray) -> tuple[NDArray, NDArray]:
    """Ensure cells are sorted by id."""
    if not np.all(cell_ids[:-1] < cell_ids[1:]):
        sort_indices = np.argsort(cell_ids)
        cell_ids = cell_ids[sort_indices]
        cell_data = cell_data[sort_indices]

    return cell_ids, cell_data


def pcolormesh(
    cell_ids: NDArray,
    cell_data: NDArray,
    healpix_grid: HealpixGrid | Mapping,
    level_tile: int | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: Colormap | str | None = None,
    norm: Norm | str | None = None,
    colorizer: Colorizer | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> list[QuadMesh]:
    """Plot data using pcolormesh.

    In nested indexing, data from the same parent cell can easily be reindexed in 2D.
    This function takes advantage of this to chunk data into larger tiles and plot each
    one of them with a :func:`~matplotlib.pyplot.pcolormesh` call.

    Parameters
    ----------
    cell_ids:
        The cell ids describing the spatial position of the data.
    cell_data:
        The data to plot with shapes:
        - (N): Scalar data mapped to colors
        - (N, 3): RGB values (0-1 float)
        - (N, 4): RGBA values (0-1 float)
    healpix_grid:
        Object or dictionnary giving parameters of the input data Healpix grid.
    level_tile:
        Healpix level/depth of tiles. If left to None, will default to `level - 11`,
        giving tiles of size 2048 x 2048.
    vmin, vmax:
        When no explicit norm is passed, `vmin` and `vmax` will be used to define the
        data range that the colormap covers.
    cmap:
       The Colormap instance or registered colormap name used to map scalar data to
       colors.
    norm:
        Normalization method used to scale data to the [0, 1] range before mapping to
        colors. See documentation of :func:`~matplotlib.pyplot.pcolormesh` for details.
    colorizer:
        The Colorizer object used to map color to data. If None, a Colorizer object is
        created from a norm and cmap.
    ax:
        Axes instance to plot onto.
    kwargs:
        Other arguments passed to `pcolormesh`.
    """
    if isinstance(healpix_grid, Mapping):
        healpix_grid = HealpixGrid(**healpix_grid)

    if healpix_grid.indexing_scheme != "nested":
        raise KeyError("Only nested indexing scheme is supported.")

    # We do some indexing work that will fail if dtypes are mismatched
    if cell_ids.dtype.kind == "i":
        cell_ids = cell_ids.astype("u8")

    cell_ids, cell_data = _to_sorted(cell_ids, cell_data)

    level = healpix_grid.level
    if level_tile is None:
        level_tile = max(0, level - 11)
    if level_tile >= level:
        raise ValueError(f"Tile level ({level_tile}) should be coarser than data level ({level})")

    if ax is None:
        ax = plt.gca()
    if isinstance(ax, GeoAxes):
        kwargs.setdefault("transform", ccrs.PlateCarree())

    if cell_data.ndim == 1:
        # We need to ensure vmin is set (either via kwargs or via the norm) otherwise
        # each tile would have a different range
        if colorizer is None:
            colorizer = Colorizer(cmap=cmap, norm=norm)
        c_vmin, c_vmax = colorizer.get_clim()
        if c_vmin is None:
            c_vmin = vmin if vmin is not None else np.nanmin(cell_data)
        if c_vmax is None:
            c_vmax = vmax if vmax is not None else np.nanmax(cell_data)
        colorizer.set_clim(c_vmin, c_vmax)
        cmap = colorizer.cmap
        norm = colorizer.norm

        cell_data = colorizer.to_rgba(cell_data)
    elif cell_data.shape[1] not in [3, 4]:
        raise IndexError(f"Invalid shape for data {cell_data.shape}")

    # Size of tiles
    n_side = 2 ** (level - level_tile)
    # Build indices to reindex a tile from nested scheme to 2d
    nested2d_indices = _build_2d_indices(level - level_tile)

    # Find the parent tile and the indices to select each tile from data
    tile_ids, tile_splits = _find_tiles(cell_ids, level, level_tile)

    quadmeshes = []

    ax_extent = ax.dataLim.frozen()
    any_partial = False
    for i_tile, tile_id in enumerate(tile_ids):
        tile_cell_ids_partial, tile_cell_data_partial = _extract_tile(
            cell_ids, cell_data, tile_splits, i_tile
        )
        tile_cell_ids, tile_cell_data = _fill_tile(
            tile_id, tile_cell_ids_partial, tile_cell_data_partial, level, level_tile
        )
        tile_cell_ids_2d, tile_cell_data_2d = _reindex_tile_2d(
            tile_cell_ids, tile_cell_data, n_side, nested2d_indices
        )
        x, y = _compute_grid(tile_cell_ids_2d, n_side, healpix_grid)

        # This will be changed by pcolormesh, save it
        ignore_existing_data_limits = ax.ignore_existing_data_limits

        quadmesh = ax.pcolormesh(x, y, tile_cell_data_2d, **kwargs)
        quadmesh.set(cmap=cmap, norm=norm, **kwargs)

        if tile_cell_ids_partial.shape[0] < tile_cell_ids.size:
            any_partial = True
            tile_extent = find_extent(tile_cell_ids_partial, healpix_grid)
            ax_extent.update_from_data_xy(
                tile_extent.get_points(), ignore=ignore_existing_data_limits
            )
            ax.ignore_existing_data_limits = False
            quadmesh.sticky_edges.x[:] = tile_extent.intervalx
            quadmesh.sticky_edges.y[:] = tile_extent.intervaly

        quadmeshes.append(quadmesh)

    if any_partial:
        if isinstance(ax, GeoAxes):
            ax.set_extent((*ax_extent.intervalx, *ax_extent.intervaly))
        else:
            ax.dataLim.set_points(ax_extent.get_points())
            ax.autoscale(enable=None)  # type: ignore[arg-type]

    return quadmeshes
