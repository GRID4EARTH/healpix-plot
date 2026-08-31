# The plot() function

{func}`healpix_plot.plot` is the main entry point of the library. It resamples HEALPix data onto a regular pixel grid and renders it with Matplotlib and Cartopy.

`plot()` returns a `matplotlib.image.AxesImage` (the mappable). Use `.axes` to access the underlying `Axes` object.

| Parameter          | Description                                                                                                                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cell_ids`         | `numpy.ndarray` of cell ids describing spatial positions.                                                                                                                                     |
| `data`             | 1-D array for scalar data (color-mapped), or 2-D array of shape `(N, 3)` / `(N, 4)` for RGB / RGBA.                                                                                           |
| `healpix_grid`     | A `HealpixGrid` instance (or equivalent dict).                                                                                                                                                |
| `sampling_grid`    | Target raster resolution and extent. Pass a dict such as `{"shape": 1024}`; missing `center` / `resolution` are inferred from the data.                                                       |
| `projection`       | A Cartopy CRS name (e.g. `"Mollweide"`) or an actual CRS object. Unknown names raise a `ValueError`.                                                                                          |
| `agg`              | Aggregation function applied when `cell_ids` contains duplicates before resampling. Accepted values: `"mean"` (default), `"median"`, `"std"`, `"var"`, `"min"`, `"max"`, `"first"`, `"last"`. |
| `interpolation`    | Resampling method. `"nearest"` (default and only implemented). `"bilinear"` is accepted by the API but raises `NotImplementedError`.                                                          |
| `background_value` | Fill value for grid points with no matching cell id. Default: `numpy.nan`.                                                                                                                    |
| `view`             | Optional `(xmin, xmax, ymin, ymax)` tuple defining the displayed extent, passed to `ax.set_extent()`. Ignored when the full sphere is being plotted.                                          |
| `rgb_clip`         | `(min, max)` tuple used to clip RGB/RGBA `data` before display. Default: `(0.0, 1.0)`. Ignored for scalar (color-mapped) data.                                                                |
| `ax`               | An existing Cartopy **GeoAxes** to draw on — see [Plot on an existing axis](#plot-on-an-existing-axis) below for the exact requirement. If omitted, a new figure is created using `projection`. |
| `title`            | Optional string title for the axes.                                                                                                                                                           |
| `cmap`             | Colormap (name or `Colormap` object). Default: `"viridis"`.                                                                                                                                   |
| `vmin`, `vmax`     | Scalar data range for colour normalisation.                                                                                                                                                   |
| `norm`             | A `matplotlib.colors.Normalize` instance for finer colour control.                                                                                                                            |
| `colorbar`         | `True` to add a colorbar, or a dict of kwargs forwarded to `figure.colorbar()`. Default: `False`.                                                                                             |
| `axis_labels`      | `None` (default, uses `"Longitude"` / `"Latitude"`), a dict with `"x"` / `"y"` keys, or `"none"` to suppress labels entirely.                                                                 |

## Minimal call

```python
healpix_plot.plot(
    cell_ids,  # 1-D array of HEALPix cell IDs (uint64)
    data,  # 1-D scalar array, or (N, 3)/(N, 4) for RGB/RGBA
    healpix_grid=grid,  # HealpixGrid object (or dict)
    sampling_grid={"shape": 1024},
)
```

## Interpolation

```python
healpix_plot.plot(..., interpolation="nearest")  # default
healpix_plot.plot(
    ..., interpolation="bilinear"
)  # smoother, better for continuous fields
```

(plot-on-an-existing-axis)=

## Plot on an existing axis

When you pass `ax`, the `projection` parameter is ignored — `plot()` draws
directly onto the axis you give it instead of creating one.

```{important}
The axis you pass as `ax` **must** be a Cartopy `GeoAxes`, i.e. created with
`subplot_kw={"projection": <a cartopy CRS>}`. `plot()` calls Cartopy-only
methods on it internally (`ax.set_extent()`, `ax.set_global()`). A plain
`matplotlib.axes.Axes` — what you get from `plt.subplots()` **without**
`subplot_kw` — does not have these methods and raises:

    AttributeError: 'Axes' object has no attribute 'set_extent'

This is easy to miss when you go from a single plot to a grid of subplots,
since `subplot_kw` has to be repeated on every `plt.subplots()` call that
will host a `plot()` call.
```

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
healpix_plot.plot(..., ax=ax)
plt.show()
```

### Multiple panels side by side

`subplot_kw` applies to every axis `plt.subplots()` creates, including when
you ask for several panels at once — pass `ax=axes[0]`, `ax=axes[1]`, etc.
to place one `plot()` call per panel:

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    subplot_kw={"projection": ccrs.Mollweide()},  # required on every axis
    figsize=(16, 8),
    layout="constrained",
)

m1 = healpix_plot.plot(
    cell_ids_a, data_a, healpix_grid=grid, sampling_grid={"shape": 512},
    ax=ax1, title="A", colorbar=True,
)
m2 = healpix_plot.plot(
    cell_ids_b, data_b, healpix_grid=grid, sampling_grid={"shape": 512},
    ax=ax2, title="B", colorbar=True,
)

plt.show()
```

Each axis can even use a different projection, since it is set per-axis via
`subplot_kw` and not by `plot()` itself when `ax` is given.

## Title and axis labels

```python
healpix_plot.plot(
    ...,
    title="Surface temperature",
    axis_labels={"x": "Longitude", "y": "Latitude"},
)
healpix_plot.plot(..., axis_labels="none")  # hide labels
```

## RGB / RGBA data

`plot()` accepts multi-band data directly. If `data` has shape `(N, 3)` (RGB) or `(N, 4)` (RGBA), the values are composited per-pixel rather than colour-mapped. Colourmap parameters (`cmap`, `vmin`, `vmax`, `norm`) are ignored in this mode. Values are clipped to `rgb_clip` (default `(0.0, 1.0)`) before display.

```python
rgb = np.stack([r, g, b], axis=1)  # shape (N, 3)
healpix_plot.plot(
    cell_ids, rgb, healpix_grid=healpix_grid, sampling_grid={"shape": 1024},
    rgb_clip=(0.0, 1.0),
)
```
