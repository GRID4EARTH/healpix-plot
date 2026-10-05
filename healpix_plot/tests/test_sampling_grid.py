import numpy as np
import pytest
from affine import Affine

from healpix_plot import sampling_grid as sg
from healpix_plot.healpix import HealpixGrid


@pytest.mark.parametrize(
    [
        "cell_ids",
        "params",
        "shape",
        "resolution",
        "center",
        "expected_resolution",
        "expected_center",
    ],
    (
        pytest.param(
            np.arange(4 * 4**3, 5 * 4**3),
            {"level": 3, "indexing_scheme": "nested", "ellipsoid": "sphere"},
            (15, 15),
            None,
            None,
            (5.625, 5.0979),
            (0, 0),
            id="base_cell4-res-center",
        ),
        pytest.param(
            np.arange(5 * 4**3, 6 * 4**3),
            {"level": 3, "indexing_scheme": "nested", "ellipsoid": "sphere"},
            (15, 15),
            None,
            None,
            (5.625, 5.0979),
            (90, 0),
            id="base_cell5-res-center",
        ),
        pytest.param(
            None,
            None,
            (15, 7),
            (0.5, 1.2),
            (9.5, 10.0),
            (0.5, 1.2),
            (9.5, 10.0),
        ),
        pytest.param(
            None,
            None,
            (7, 5),
            None,
            (10.0, 10.0),
            (1.875, 3.590377890729141),
            (10.0, 10.0),
        ),
        pytest.param(
            None,
            None,
            (7, 5),
            (0.5, 0.5),
            None,
            (0.5, 0.5),
            (11.25, 0.0),
        ),
    ),
)
def test_infer_parameters(
    cell_ids, params, shape, resolution, center, expected_resolution, expected_center
):
    if params is None:
        healpix_params = HealpixGrid(
            level=4, indexing_scheme="nested", ellipsoid="sphere"
        )
    else:
        healpix_params = HealpixGrid(**params)
    if cell_ids is None:
        cell_ids = np.array(
            [
                1079,
                1099,
                1120,
                1121,
                1122,
                1123,
                1124,
                1126,
                1127,
                1128,
                1129,
                1131,
                1132,
                1133,
                1134,
                1135,
                1144,
                1220,
            ],
            dtype="uint64",
        )

    grid = sg.ParametrizedSamplingGrid(
        shape=shape, resolution=resolution, center=center
    )
    actual_shape, actual_resolution, actual_center = sg._infer_parameters(
        grid, cell_ids, healpix_params
    )

    assert actual_shape == shape
    assert actual_resolution == pytest.approx(expected_resolution)
    assert actual_center == pytest.approx(expected_center)


class TestParametrizedSamplingGrid:
    @pytest.mark.parametrize(
        ["shape", "expected_shape"], ((3, (3, 3)), ((2, 5), (2, 5)))
    )
    @pytest.mark.parametrize(
        ["resolution", "expected_resolution"],
        ((1.1, (1.1, 1.1)), ((0.5, 1.2), (0.5, 1.2)), (None, None)),
    )
    @pytest.mark.parametrize(
        ["center", "expected_center"],
        (((10.1, 13.6), (10.1, 13.6)), ((0.5, 1.2), (0.5, 1.2)), (None, None)),
    )
    def test_from_parameters(
        self,
        shape,
        expected_shape,
        resolution,
        expected_resolution,
        center,
        expected_center,
    ):
        actual = sg.ParametrizedSamplingGrid.from_parameters(
            shape=shape, resolution=resolution, center=center
        )

        assert actual.shape == expected_shape
        assert actual.resolution == expected_resolution
        assert actual.center == expected_center

    def test_from_dict(self):
        shape = (3, 4)
        resolution = (0.5, 1.0)
        center = (45.3, -4.3)
        actual = sg.ParametrizedSamplingGrid.from_dict(
            {"shape": shape, "resolution": resolution, "center": center}
        )

        assert actual.shape == shape
        assert actual.resolution == resolution
        assert actual.center == center

    @pytest.mark.parametrize(
        ["bbox", "shape", "expected_resolution", "expected_center"],
        (
            ((0, 0, 5, 5), 10, (5 / 9, 5 / 9), (2.5, 2.5)),
            ((0, -5, 10, 15), (11, 21), (1.0, 1.0), (5, 5)),
        ),
    )
    def test_from_bbox(self, bbox, shape, expected_resolution, expected_center):
        actual = sg.ParametrizedSamplingGrid.from_bbox(bbox, shape)
        assert actual.shape == (shape if isinstance(shape, tuple) else (shape, shape))
        assert actual.resolution == expected_resolution
        assert actual.center == expected_center

    @pytest.mark.parametrize(
        ["shape", "resolution", "center", "expected"],
        (
            pytest.param(
                (5, 5),
                (1, 1),
                (0, 0),
                np.stack(
                    np.meshgrid(
                        0.0 + 1.0 * np.arange(-2, 3),
                        0.0 + 1.0 * np.arange(-2, 3),
                    ),
                    axis=0,
                )[::-1, ...],
            ),
            pytest.param(
                (7, 5),
                (0.5, 0.5),
                (180, 0),
                np.stack(
                    np.meshgrid(
                        180 + 0.5 * np.arange(-3, 4),
                        0.5 * np.arange(-2, 3),
                    ),
                    axis=0,
                )[::-1, ...],
            ),
            pytest.param(
                (15, 7),
                (0.5, 1.2),
                (9.5, 10.0),
                np.stack(
                    np.meshgrid(
                        9.5 + 0.5 * np.arange(-7, 8),
                        10.0 + 1.2 * np.arange(-3, 4),
                    ),
                    axis=0,
                )[::-1, ...],
            ),
        ),
    )
    def test_resolve(self, shape, resolution, center, expected):
        params = HealpixGrid(level=4, indexing_scheme="nested", ellipsoid="WGS84")
        cell_ids = np.array(
            [
                1079,
                1099,
                1120,
                1121,
                1122,
                1123,
                1124,
                1126,
                1127,
                1128,
                1129,
                1131,
                1132,
                1133,
                1134,
                1135,
                1144,
                1220,
            ],
            dtype="uint64",
        )
        grid = sg.ParametrizedSamplingGrid(
            shape=shape, resolution=resolution, center=center
        )
        actual = grid.resolve(cell_ids, params)
        expected_y, expected_x = expected
        np.testing.assert_allclose(actual.x, expected_x)
        np.testing.assert_allclose(actual.y, expected_y)


class TestAffineSamplingGrid:
    @pytest.mark.parametrize(
        ["transform", "shape", "expected_shape"],
        (
            (Affine.translation(1, 1), 3, (3, 3)),
            (Affine.scale(2, 0.5), (10, 15), (10, 15)),
        ),
    )
    def test_from_transform(self, transform, shape, expected_shape):
        actual = sg.AffineSamplingGrid.from_transform(transform, shape)
        assert actual.transform == transform
        assert actual.shape == expected_shape
        assert actual.crs is None
        assert actual.is_geographic

    @pytest.mark.parametrize(
        ["transform", "shape", "expected_x", "expected_y"],
        (
            pytest.param(
                Affine.translation(1, 1),
                (3, 3),
                np.broadcast_to(np.arange(3) + 1.5, (3, 3)),
                np.broadcast_to((np.arange(3) + 1.5)[:, None], (3, 3)),
                id="translation",
            ),
            pytest.param(
                Affine.scale(2, 1),
                (4, 5),
                np.broadcast_to((np.arange(5) + 0.5) * 2, (4, 5)),
                np.broadcast_to((np.arange(4) + 0.5)[:, None], (4, 5)),
                id="scale",
            ),
            pytest.param(
                Affine(0.5, 0, 10, 0, -0.5, 20),
                (2, 3),
                np.broadcast_to(np.array([10.25, 10.75, 11.25]), (2, 3)),
                np.broadcast_to(np.array([[19.75], [19.25]]), (2, 3)),
                id="north-up",
            ),
        ),
    )
    def test_resolve(self, transform, shape, expected_x, expected_y):
        grid = sg.AffineSamplingGrid(transform, shape)

        # the parameters are ignored
        actual = grid.resolve(0, None)
        assert actual.shape == shape
        assert actual.is_axis_aligned
        assert actual.crs is None
        assert actual.transform == transform
        np.testing.assert_allclose(actual.x, expected_x)
        np.testing.assert_allclose(actual.y, expected_y)

    def test_extent_and_origin(self):
        north_up = sg.AffineSamplingGrid(Affine(0.5, 0, 10, 0, -0.5, 20), (2, 3))
        actual = north_up.resolve(0, None)
        assert actual.extent == (10, 11.5, 19, 20)
        assert actual.origin == "upper"

        south_up = sg.AffineSamplingGrid(Affine(1, 0, 0, 0, 1, 0), (2, 3))
        actual = south_up.resolve(0, None)
        assert actual.extent == (0, 3, 0, 2)
        assert actual.origin == "lower"

    def test_rotated(self):
        transform = Affine.rotation(45) @ Affine.scale(2)
        grid = sg.AffineSamplingGrid(transform, (3, 4))
        actual = grid.resolve(0, None)

        assert not actual.is_axis_aligned
        corner_x, corner_y = actual.corners()
        assert corner_x.shape == (4, 5)
        assert corner_y.shape == (4, 5)
        # the extent is the bounding box of the corners
        assert actual.extent == pytest.approx(
            (corner_x.min(), corner_x.max(), corner_y.min(), corner_y.max())
        )

    def test_projected_crs(self):
        pyproj = pytest.importorskip("pyproj")

        grid = sg.AffineSamplingGrid(
            Affine(1000, 0, 0, 0, -1000, 2000), (2, 2), crs="EPSG:3857"
        )
        assert isinstance(grid.crs, pyproj.CRS)
        assert not grid.is_geographic

        actual = grid.resolve(0, None)
        assert not actual.is_geographic
        # pixel centres: x = 500, 1500; y = 1500, 500
        expected_lon = np.degrees(np.array([500, 1500]) / 6378137)
        np.testing.assert_allclose(actual.lon, np.broadcast_to(expected_lon, (2, 2)))
        assert np.all(actual.lat[0, :] > actual.lat[1, :])
        assert actual.valid.all()

    def test_geographic_crs(self):
        grid = sg.AffineSamplingGrid(Affine(1, 0, 0, 0, 1, 0), (2, 2), crs="EPSG:4326")
        assert grid.is_geographic
        actual = grid.resolve(0, None)
        np.testing.assert_equal(actual.lon, actual.x)
        np.testing.assert_equal(actual.lat, actual.y)

    def test_pixel_indices(self):
        healpix_grid = HealpixGrid(
            level=1, indexing_scheme="nested", ellipsoid="sphere"
        )
        # 1° pixels, north-up, covering the globe
        grid = sg.AffineSamplingGrid(Affine(1, 0, -180, 0, -1, 90), (180, 360))
        cell_ids = np.arange(48, dtype="uint64")

        rows, cols = grid.pixel_indices(cell_ids, healpix_grid)

        lon, lat = healpix_grid.operations.healpix_to_lonlat(
            cell_ids, **healpix_grid.as_keyword_params()
        )
        np.testing.assert_equal(rows, np.floor(90 - lat))
        np.testing.assert_equal(cols, np.floor(lon + 180))

    def test_from_raster(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        pyproj = pytest.importorskip("pyproj")

        transform = Affine(1000, 0, 0, 0, -1000, 2000)
        path = tmp_path / "raster.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=3,
            height=2,
            count=1,
            dtype="float32",
            crs="EPSG:3857",
            transform=transform,
        ) as dst:
            dst.write(np.zeros((2, 3), dtype="float32"), 1)

        with rasterio.open(path) as src:
            grid = sg.AffineSamplingGrid.from_raster(src)
        assert grid.transform == transform
        assert grid.shape == (2, 3)
        assert grid.crs == pyproj.CRS("EPSG:3857")

        rioxarray = pytest.importorskip("rioxarray")
        arr = rioxarray.open_rasterio(path)
        grid = sg.AffineSamplingGrid.from_raster(arr)
        assert grid.transform.almost_equals(transform)
        assert grid.shape == (2, 3)
        assert grid.crs == pyproj.CRS("EPSG:3857")
