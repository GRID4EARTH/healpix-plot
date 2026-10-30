from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Self, TypedDict

import healpix_geo.auto
import numpy as np
from affine import Affine
from healpix_geo.auto import Grid
from numpy.typing import NDArray


class BoxSamplingGridDict(TypedDict):
    """Parameters for BoxSamplingGrid."""

    shape: int | tuple[int, int] | None
    resolution: float | tuple[float, float] | None
    extents: tuple[float, float, float, float] | None


class SamplingGrid:
    """Abstract class of sampling grid."""

    def resolve(
        self, cell_ids: NDArray[np.integer], healpix_grid: Grid
    ) -> ConcreteSamplingGrid:  # pragma: no cover
        """Return a concrete sampling grid."""
        raise NotImplementedError


def crosses_prime_meridian(cell_ids, params):
    # very coarse detection of prime meridian crossing
    base_cells = cell_ids // 4**params.level
    crossing_base_cells = np.array([0, 3, 4, 8, 11], dtype="uint64")
    return np.any(np.isin(crossing_base_cells, base_cells))


class BoxSamplingGrid(SamplingGrid):
    def __init__(
        self,
        extents: tuple[float, float, float, float] | None = None,
        shape: int | tuple[int, int] | None = None,
        resolution: float | tuple[float, float] | None = None,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape, shape)
        if isinstance(resolution, float):
            resolution = (resolution, resolution)

        self.shape = shape
        self.extents = extents
        self.resolution = resolution

    def resolve(
        self,
        cell_ids: NDArray[np.integer],
        healpix_grid: Grid,
    ) -> ConcreteSamplingGrid:
        """Return a concrete grid. Guess missing parameters from HEALPix cells."""
        # We want extents + shape

        if self.shape is not None and self.resolution is not None:
            warnings.warn(
                "Both 'shape' and 'resolution' were specified. "
                "Resolution will be ignored."
            )

        if self.extents is None:
            lon, lat = healpix_geo.auto.vertices(cell_ids, healpix_grid)
            if crosses_prime_meridian(cell_ids, healpix_grid):
                lon = (lon + 180) % 360 - 180

            xmin = np.min(lon[..., 3])
            xmax = np.max(lon[..., 1])
            ymin = np.min(lat[..., 0])
            ymax = np.max(lat[..., 2])
            self.extents = xmin, ymin, xmax, ymax

        xmin, ymin, xmax, ymax = self.extents
        if self.shape is None:
            if self.resolution is None:
                raise ValueError("Specify at least one of 'resolution' or 'shape'.")
            size_x = int((xmax - xmin) / self.resolution[0])
            size_y = int((ymax - ymin) / self.resolution[1])
            self.shape = (size_x, size_y)

        size_x, size_y = self.shape
        x = np.linspace(xmin, xmax, size_x)
        y = np.linspace(ymin, ymax, size_y)

        xx, yy = np.meshgrid(x, y)
        return ConcreteSamplingGrid(xx, yy, (xmin, xmax), (ymin, ymax))


@dataclass
class AffineSamplingGrid(SamplingGrid):
    transform: Affine
    shape: tuple[int, int]

    @classmethod
    def from_transform(
        cls,
        transform: Affine,
        shape: int | tuple[int, int],
    ) -> Self:
        if isinstance(shape, int):
            shape = (shape, shape)

        return cls(transform, shape)

    @property
    def center_transform(self):
        return self.transform

    @property
    def corner_transform(self):
        return self.transform * Affine.translation(-0.5, -0.5)

    def resolve(self, cell_ids: np.ndarray, parameters: Grid) -> ConcreteSamplingGrid:
        pixel_x = np.arange(self.shape[0])
        pixel_y = np.arange(self.shape[1])

        py, px = np.meshgrid(pixel_y, pixel_x)

        x, y = self.center_transform * (px, py)

        _, scale_x, _, _, _, scale_y = self.center_transform.to_gdal()

        (xmin, xmax), (ymin, ymax) = self.center_transform * (
            np.array([0, self.shape[0]]),
            np.array([0, self.shape[1]]),
        )

        if xmin <= 0 and xmax >= 0:
            x = (x + 180) % 360 - 180

        extent_x = (float(xmin), float(xmax))
        extent_y = (float(ymin), float(ymax))

        return ConcreteSamplingGrid(x, y, extent_x, extent_y)


@dataclass
class ConcreteSamplingGrid:
    x: np.ndarray
    y: np.ndarray

    extent_x: tuple[float, float]
    extent_y: tuple[float, float]

    @property
    def shape(self):
        return self.x.shape

    @property
    def extent(self):
        extent_x = tuple((x + 180) % 360 - 180 for x in self.extent_x)
        return extent_x + self.extent_y
