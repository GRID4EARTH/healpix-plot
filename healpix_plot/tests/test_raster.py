import numpy as np
import pytest

from healpix_plot.healpix import HealpixGrid
from healpix_plot.raster import raster_cell_ids, to_dataarray
from healpix_plot.resampling import resample
from healpix_plot.sampling_grid import AffineSamplingGrid, ParametrizedSamplingGrid


def test_to_dataarray_axis_aligned():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("pyproj")

    healpix_grid = HealpixGrid(level=2, indexing_scheme="nested", ellipsoid="sphere")
    cell_ids = np.arange(12 * 4**2, dtype="uint64")
    data = np.ones(cell_ids.size)

    grid = ParametrizedSamplingGrid.from_bbox((-20, -10, 20, 10), shape=(9, 5))
    target, image = resample(
        cell_ids,
        data,
        sampling_grid=grid,
        healpix_grid=healpix_grid,
        interpolation="nearest",
        agg="first",
    )
    arr = to_dataarray(image, target, healpix_grid, name="ones")

    assert isinstance(arr, xr.DataArray)
    assert arr.dims == ("y", "x")
    assert arr.name == "ones"
    np.testing.assert_allclose(arr.x, np.linspace(-20, 20, 9))
    np.testing.assert_allclose(arr.y, np.linspace(-10, 10, 5))

    pytest.importorskip("rioxarray")
    assert arr.rio.crs is not None
    assert arr.rio.crs.is_geographic


def test_geotiff_roundtrip(tmp_path):
    pytest.importorskip("pyproj")
    rasterio = pytest.importorskip("rasterio")
    rioxarray = pytest.importorskip("rioxarray")
    import pyproj

    healpix_grid = HealpixGrid(level=2, indexing_scheme="nested", ellipsoid="WGS84")
    npix = 12 * 4**2
    cell_ids = np.arange(npix, dtype="uint64")
    data = cell_ids.astype("float64")

    grid = AffineSamplingGrid.from_healpix(healpix_grid)
    target, image = resample(
        cell_ids,
        data,
        sampling_grid=grid,
        healpix_grid=healpix_grid,
        interpolation="nearest",
        agg="first",
    )
    arr = to_dataarray(image, target, healpix_grid, name="cell_ids")
    assert arr.rio.crs == grid.crs
    assert arr.rio.transform().almost_equals(grid.transform)

    path = tmp_path / "healpix.tif"
    arr.rio.write_nodata(np.nan, inplace=True)
    arr.rio.to_raster(path)

    # rasterio sees the same CRS and transform ...
    with rasterio.open(path) as src:
        assert pyproj.CRS.from_user_input(src.crs) == grid.crs
        assert src.transform.almost_equals(grid.transform)
        assert (src.height, src.width) == grid.shape
        read = src.read(1)
        ids = raster_cell_ids(src, healpix_grid)

    # ... and the cell ids can be recovered from the pixel positions alone
    valid = ~np.ma.getmaskarray(ids)
    assert valid.sum() == npix
    np.testing.assert_equal(np.isfinite(read), valid)
    np.testing.assert_equal(read[valid].astype("uint64"), np.ma.getdata(ids)[valid])

    # the same through rioxarray
    reopened = rioxarray.open_rasterio(path).squeeze("band", drop=True)
    ids2 = raster_cell_ids(reopened, healpix_grid)
    np.testing.assert_equal(np.ma.getmaskarray(ids2), ~valid)
    np.testing.assert_equal(np.ma.getdata(ids2)[valid], np.ma.getdata(ids)[valid])

    # GeoTIFF has no GeoKeys for the HEALPix projection: GDAL stores the CRS in
    # a side-car. Without it, the CRS is recovered from the WKT tag.
    sidecar = path.with_name(path.name + ".aux.xml")
    if sidecar.exists():
        sidecar.unlink()
    with rasterio.open(path) as src:
        assert src.tags()["healpix_crs_wkt"]
        assert src.tags()["healpix_depth"] == "2"
        assert src.tags()["healpix_ellipsoid"] == "WGS84"
        assert AffineSamplingGrid.from_raster(src).crs == grid.crs
        ids3 = raster_cell_ids(src, healpix_grid)
    np.testing.assert_equal(ids3, ids)
    reopened = rioxarray.open_rasterio(path).squeeze("band", drop=True)
    assert AffineSamplingGrid.from_raster(reopened).crs == grid.crs
