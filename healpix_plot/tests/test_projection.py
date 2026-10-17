import numpy as np
import pytest

from healpix_plot import projection
from healpix_plot.healpix import HealpixGrid
from healpix_plot.resampling import resample
from healpix_plot.sampling_grid import AffineSamplingGrid


class TestResolveEllipsoid:
    def test_names(self):
        assert projection.resolve_ellipsoid("sphere") == {
            "name": "sphere",
            "radius": 6370997.0,
        }
        assert projection.resolve_ellipsoid(None) == projection.resolve_ellipsoid(
            "sphere"
        )
        wgs84 = projection.resolve_ellipsoid("WGS84")
        assert wgs84["semimajor_axis"] == 6378137.0
        assert wgs84["inverse_flattening"] == pytest.approx(298.257223563)

    def test_mappings_and_objects(self):
        mapping = {"semimajor_axis": 1.0, "inverse_flattening": 300.0}
        assert projection.resolve_ellipsoid(mapping) == mapping

        class Ellipsoid:
            semimajor_axis = 2.0
            inverse_flattening = 299.0

        assert projection.resolve_ellipsoid(Ellipsoid()) == {
            "semimajor_axis": 2.0,
            "inverse_flattening": 299.0,
        }

        class Sphere:
            radius = 3.0

        assert projection.resolve_ellipsoid(Sphere()) == {"radius": 3.0}

    def test_grid_like(self):
        grid = HealpixGrid(level=1, indexing_scheme="nested", ellipsoid="WGS84")
        assert projection.resolve_ellipsoid(grid) == projection.resolve_ellipsoid(
            "WGS84"
        )


def test_proj_string():
    assert projection.proj_string("sphere") == "+proj=healpix +R=6370997.0 +lon_0=0.0"
    assert (
        projection.proj_string("WGS84", lon_0=90)
        == "+proj=healpix +a=6378137.0 +rf=298.257223563 +lon_0=90.0"
    )
    assert projection.geographic_proj_string("sphere") == "+proj=longlat +R=6370997.0"


def test_crs():
    pytest.importorskip("pyproj")

    crs = projection.healpix_crs("WGS84")
    assert crs.is_projected
    assert crs.ellipsoid.semi_major_metre == 6378137.0

    geographic = projection.geographic_crs("WGS84")
    assert geographic.is_geographic
    assert geographic.ellipsoid.semi_major_metre == 6378137.0

    # the plane is 2πR wide, R being the authalic radius
    half_width = projection.plane_half_width(crs)
    assert half_width == pytest.approx(np.pi * 6371007.18, rel=1e-6)


@pytest.mark.parametrize("ellipsoid", ["sphere", "WGS84"])
@pytest.mark.parametrize("indexing_scheme", ["nested", "ring"])
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_cell_aligned_grid_is_lossless(level, indexing_scheme, ellipsoid):
    pytest.importorskip("pyproj")

    healpix_grid = HealpixGrid(
        level=level, indexing_scheme=indexing_scheme, ellipsoid=ellipsoid
    )
    nside = 2**level
    npix = 12 * nside**2

    grid = AffineSamplingGrid.from_healpix(healpix_grid)
    assert grid.shape == (6 * nside, 6 * nside)
    assert not grid.is_geographic

    target = grid.resolve(None, healpix_grid)
    assert not target.is_axis_aligned

    # pixel -> cell: every cell is hit by exactly one pixel centre
    ids = target.cell_ids(healpix_grid)
    found = np.ma.compressed(ids)
    assert found.size == npix
    np.testing.assert_equal(np.sort(found), np.arange(npix))

    # cell -> pixel agrees with pixel -> cell
    rows, cols = grid.pixel_indices(np.arange(npix), healpix_grid)
    assert np.all((rows >= 0) & (rows < grid.shape[0]))
    assert np.all((cols >= 0) & (cols < grid.shape[1]))
    np.testing.assert_equal(np.ma.getdata(ids)[rows, cols], np.arange(npix))


def test_cell_aligned_grid_crop():
    pytest.importorskip("pyproj")

    healpix_grid = HealpixGrid(level=3, indexing_scheme="nested", ellipsoid="WGS84")
    # the children of base cell 4 (equatorial, centred on lon 0)
    cell_ids = np.arange(4 * 4**3, 5 * 4**3, dtype="uint64")

    full = AffineSamplingGrid.from_healpix(healpix_grid)
    grid = AffineSamplingGrid.from_healpix(healpix_grid, cell_ids)
    assert grid.shape == (8, 8)
    assert grid.shape < full.shape
    assert grid.crs == full.crs

    ids = grid.resolve(None, healpix_grid).cell_ids(healpix_grid)
    np.testing.assert_equal(np.sort(np.ma.compressed(ids)), cell_ids)


def test_cell_aligned_grid_lon_0():
    pytest.importorskip("pyproj")

    healpix_grid = HealpixGrid(level=2, indexing_scheme="nested", ellipsoid="sphere")

    grid = AffineSamplingGrid.from_healpix(healpix_grid, lon_0=90)
    ids = grid.resolve(None, healpix_grid).cell_ids(healpix_grid)
    np.testing.assert_equal(np.sort(np.ma.compressed(ids)), np.arange(12 * 4**2))

    with pytest.raises(ValueError, match="multiple of 90"):
        AffineSamplingGrid.from_healpix(healpix_grid, lon_0=45)


def test_cell_aligned_grid_requires_level_1():
    pytest.importorskip("pyproj")

    healpix_grid = HealpixGrid(level=0, indexing_scheme="nested", ellipsoid="sphere")
    with pytest.raises(ValueError, match="level 1 or higher"):
        AffineSamplingGrid.from_healpix(healpix_grid)


def test_cell_aligned_grid_requires_single_level():
    pytest.importorskip("pyproj")

    healpix_grid = HealpixGrid(level=2, indexing_scheme="zuniq", ellipsoid="sphere")
    with pytest.raises(ValueError, match="single resolution level"):
        AffineSamplingGrid.from_healpix(healpix_grid)


def test_resample_on_cell_aligned_grid_is_lossless():
    pytest.importorskip("pyproj")

    healpix_grid = HealpixGrid(level=2, indexing_scheme="nested", ellipsoid="WGS84")
    npix = 12 * 4**2
    cell_ids = np.arange(npix, dtype="uint64")
    data = np.arange(npix, dtype="float64") * 10

    grid = AffineSamplingGrid.from_healpix(healpix_grid)
    target, image = resample(
        cell_ids,
        data,
        sampling_grid=grid,
        healpix_grid=healpix_grid,
        interpolation="nearest",
        agg="first",
    )

    assert image.shape == grid.shape
    finite = np.isfinite(image)
    assert finite.sum() == npix
    np.testing.assert_equal(np.sort(image[finite]), data)
    rows, cols = grid.pixel_indices(cell_ids, healpix_grid)
    np.testing.assert_equal(image[rows, cols], data)
