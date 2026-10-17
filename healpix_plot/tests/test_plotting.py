import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from affine import Affine  # noqa: E402

from healpix_plot import HEALPix, HealpixGrid, plot  # noqa: E402
from healpix_plot.sampling_grid import AffineSamplingGrid  # noqa: E402


@pytest.fixture
def healpix_data():
    healpix_grid = HealpixGrid(level=2, indexing_scheme="nested", ellipsoid="WGS84")
    cell_ids = np.arange(12 * 4**2, dtype="uint64")
    lon, lat = healpix_grid.operations.healpix_to_lonlat(
        cell_ids, **healpix_grid.as_keyword_params()
    )
    data = np.cos(np.deg2rad(lon)) * np.sin(np.deg2rad(lat))
    return healpix_grid, cell_ids, data


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestHEALPix:
    def test_outline(self):
        projection = HEALPix(ellipsoid="WGS84")
        xmin, xmax = projection.x_limits
        ymin, ymax = projection.y_limits
        assert xmax == pytest.approx(-xmin)
        assert ymax == pytest.approx(-ymin)
        assert xmax == pytest.approx(2 * ymax)
        assert projection.boundary.is_ring

    def test_matches_grid_crs(self):
        pytest.importorskip("pyproj")
        from healpix_plot.projection import healpix_crs

        assert HEALPix(ellipsoid="WGS84") == ccrs.Projection(healpix_crs("WGS84"))
        assert HEALPix(ellipsoid="sphere") != ccrs.Projection(healpix_crs("WGS84"))


def test_plot_geographic_grid(healpix_data):
    healpix_grid, cell_ids, data = healpix_data

    mappable = plot(
        cell_ids, data, healpix_grid=healpix_grid, sampling_grid={"shape": 64}
    )
    assert isinstance(mappable, matplotlib.image.AxesImage)


def test_plot_projected_axis_aligned_grid(healpix_data):
    pytest.importorskip("pyproj")
    healpix_grid, cell_ids, data = healpix_data

    grid = AffineSamplingGrid(
        Affine(200_000, 0, -6_000_000, 0, -200_000, 6_000_000),
        (60, 60),
        crs="EPSG:3857",
    )
    # on axes in a different projection, cartopy re-projects the image
    mappable = plot(
        cell_ids,
        data,
        healpix_grid=healpix_grid,
        sampling_grid=grid,
        projection="Mercator",
    )
    assert isinstance(mappable, matplotlib.image.AxesImage)

    # on axes in the grid's own projection, the image is drawn as is
    ax = plt.axes(projection=ccrs.Projection(grid.crs))
    mappable = plot(
        cell_ids, data, healpix_grid=healpix_grid, sampling_grid=grid, ax=ax
    )
    assert isinstance(mappable, matplotlib.image.AxesImage)
    assert mappable.get_extent() == pytest.approx(grid.resolve(None, None).extent)


@pytest.mark.parametrize("projection", ["HEALPix", "Mollweide"])
def test_plot_cell_aligned_grid(healpix_data, projection):
    pytest.importorskip("pyproj")
    healpix_grid, cell_ids, data = healpix_data

    grid = AffineSamplingGrid.from_healpix(healpix_grid)
    mappable = plot(
        cell_ids,
        data,
        healpix_grid=healpix_grid,
        sampling_grid=grid,
        projection=projection,
    )
    assert isinstance(mappable, matplotlib.collections.QuadMesh)
    if projection == "HEALPix":
        assert isinstance(mappable.axes.projection, HEALPix)
        # drawn in data coordinates, without re-projection
        np.testing.assert_allclose(
            mappable.get_coordinates()[..., 0], grid.resolve(None, None).corners()[0]
        )
    # the whole globe is covered
    mappable.axes.coastlines()
    mappable.figure.canvas.draw()
