# Sampling grids

The sampling grid defines the regular pixel grid onto which HEALPix data is resampled before being rendered by Matplotlib. It controls the spatial extent, pixel resolution, and output image shape.

Three ways to define the target raster:

## 1. Parametrised (dict or `ParametrizedSamplingGrid`)

Pass a dict to `sampling_grid`. Recognised keys:

| Key          | Default  | Description                                                                           |
| ------------ | -------- | ------------------------------------------------------------------------------------- |
| `shape`      | `1024`   | Output array size. An `int` produces a square grid; a 2-tuple sets `(width, height)`. |
| `resolution` | inferred | Step size in degrees. A `float` expands to equal x/y steps.                           |
| `center`     | inferred | `(lon, lat)` centre of the grid in degrees.                                           |

```python
sampling_grid = {"shape": (2048, 1024), "center": (0.0, 0.0)}
```

The spatial extent and pixel resolution are inferred automatically from the bounding box of your `cell_ids`.

## 2. Bounding box (`ParametrizedSamplingGrid.from_bbox`)

Use `ParametrizedSamplingGrid.from_bbox()` to pin the output to a fixed region regardless of the data:

```python
from healpix_plot.sampling_grid import ParametrizedSamplingGrid

sampling_grid = ParametrizedSamplingGrid.from_bbox(
    bbox=(5, 36.0, 13.0, 45.0),  # (lon_min, lat_min, lon_max, lat_max) in degrees
    shape=512,
)
```

This is the right choice when comparing multiple datasets or animating over time.

## 3. Affine transform (`AffineSamplingGrid`)

Use `AffineSamplingGrid` when the output pixels must align with a reference raster (e.g. a GeoTIFF). It follows the rasterio / GDAL conventions: the transform maps pixel indices `(col, row)` to the coordinates of the _outer corner_ of the pixel, and the shape is `(height, width)`:

```python
from healpix_plot.sampling_grid import AffineSamplingGrid
from affine import Affine

transform = Affine(
    0.01, 0, 5, 0, -0.01, 45
)  # 0.01 deg/pixel, top-left corner at (5, 45)
sampling_grid = AffineSamplingGrid.from_transform(transform, shape=(2500, 4000))
```

Without a `crs`, the coordinates are longitude / latitude in degrees. Pass `crs` (anything `pyproj.CRS.from_user_input` accepts) to sample in a projected coordinate system; the pixel centres are then projected back to longitude / latitude before looking up the cells, and pixels outside the projection's domain stay at the background value:

```python
sampling_grid = AffineSamplingGrid.from_transform(
    Affine(10_000, 0, -2_000_000, 0, -10_000, 6_000_000),
    shape=(400, 400),
    crs="EPSG:3857",
)
```

The grid of an existing raster can be taken directly from an open rasterio dataset or a rioxarray-enabled `DataArray`:

```python
import rasterio

with rasterio.open("reference.tif") as src:
    sampling_grid = AffineSamplingGrid.from_raster(src)
```

### The lossless grid: one pixel per cell

In PROJ's ellipsoidal HEALPix projection (`+proj=healpix`, the same construction as healpix-geo's ellipsoidal cells) every cell of a given level is a square rotated by 45°. `AffineSamplingGrid.from_healpix` builds the rotated grid on which each pixel _is_ a cell, so resampling onto it loses nothing and the result can be written to a GeoTIFF and read back cell by cell:

```python
import healpix_plot
from healpix_plot.raster import to_dataarray, raster_cell_ids

healpix_grid = healpix_plot.HealpixGrid(
    level=6, indexing_scheme="nested", ellipsoid="WGS84"
)
sampling_grid = AffineSamplingGrid.from_healpix(healpix_grid)  # the whole globe
# sampling_grid = AffineSamplingGrid.from_healpix(healpix_grid, cell_ids)  # cropped to the data

target, image = healpix_plot.resample(
    cell_ids,
    data,
    sampling_grid=sampling_grid,
    healpix_grid=healpix_grid,
    interpolation="nearest",
    agg="first",
)
raster = to_dataarray(image, target, healpix_grid)  # georeferenced xarray.DataArray
raster.rio.to_raster("healpix.tif")  # needs rioxarray

with rasterio.open("healpix.tif") as src:
    cell_ids_per_pixel = raster_cell_ids(src, healpix_grid)  # masked off the globe
```

Things to know about this grid:

- The pixels are rotated, so the raster is drawn with `pcolormesh` rather than `imshow`; `plot(..., projection="HEALPix")` shows it undistorted.
- The CRS has no EPSG code and no GeoTIFF GeoKeys, so GDAL keeps it in a `<file>.tif.aux.xml` side-car. `to_dataarray` also writes it as WKT into the TIFF tags (`healpix_crs_wkt`, with `healpix_depth`, `healpix_indexing_scheme` and `healpix_ellipsoid`), and `from_raster` / `raster_cell_ids` fall back to that tag when the side-car is missing. Software outside the GDAL/PROJ family may read neither.
- `+lon_0` must be a multiple of 90° and `level` at least 1. The full grid is `6 * nside` pixels on each side, of which a third lie on the globe; pass `cell_ids` to crop it.
- Only `"nested"` and `"ring"` (single-level) grids are supported.
- Interpolating or resampling _across_ facet seams in this projection is wrong, as the projection is interrupted there.

:::{seealso}
See {doc}`../tutorials/quickstart` for more information
:::
