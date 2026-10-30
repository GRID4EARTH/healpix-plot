from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import healpix_geo.auto
import numpy as np
import numpy_groupies
from healpix_geo.auto import Grid
from numpy.typing import NDArray

from healpix_plot.sampling_grid import (
    BoxSamplingGrid,
    BoxSamplingGridDict,
    ConcreteSamplingGrid,
    SamplingGrid,
)

AggType = Literal["mean", "median", "std", "var", "min", "max", "first", "last"]


class Interpolator:
    """Abstract class for interpolating data from HEALPix to regular grid."""

    def __init__(
        self,
        cell_ids: NDArray[np.integer],
        healpix_grid: Grid,
        target_grid: ConcreteSamplingGrid,
        **kwargs: Any,
    ) -> None:
        self.cell_ids = cell_ids
        self.healpix_grid = healpix_grid
        self.target_grid = target_grid

    def is_rgb(self, data: np.ndarray) -> bool:
        """Whether the data is representing RGB(A)."""
        return data.ndim == 2 and data.shape[-1] in (3, 4)

    def interpolate(
        self, data: np.ndarray, background_value: float
    ) -> NDArray[np.float64]:
        """Interpolate data.

        Parameters
        ----------
        """
        raise NotImplementedError


class NearestInterpolator(Interpolator):
    """Nearest interpolation."""

    def interpolate(
        self, data: np.ndarray, background_value: float
    ) -> NDArray[np.float64]:
        target_cell_ids = healpix_geo.auto.lonlat_to_healpix(
            np.reshape(self.target_grid.x, -1),
            np.reshape(self.target_grid.y, -1),
            self.healpix_grid,
        )

        raw_shape = (target_cell_ids.size,)
        shape = self.target_grid.shape
        if self.is_rgb(data):
            raw_shape += data.shape[-1:]
            shape += data.shape[-1:]

        # indices that correspond to the target cells
        #
        # searchsorted(a, b) returns insert indices that insert a into b, not search
        # values from a in b. We thus need to mask all target cell ids not in the
        # source.
        valid = np.isin(target_cell_ids, self.cell_ids)
        indices = np.searchsorted(self.cell_ids, target_cell_ids, side="left")
        valid_indices = indices[valid]

        # actual interpolation
        image = np.full(raw_shape, fill_value=background_value, dtype=np.float64)
        image[valid, ...] = data[valid_indices, ...]

        return np.reshape(image, shape)


class BilinearInterpolator(Interpolator):
    """Bilinear interpolation. To implement."""

    pass


interpolators = {
    "nearest": NearestInterpolator,
    "bilinear": BilinearInterpolator,
}


def resample(
    cell_ids: np.ndarray,
    data: np.ndarray,
    *,
    sampling_grid: SamplingGrid | BoxSamplingGridDict,
    healpix_grid: Grid | Mapping,
    interpolation: Interpolator | Literal["nearest", "bilinear"] = "nearest",
    agg: AggType = "mean",
    background_value: float = np.nan,
) -> tuple[ConcreteSamplingGrid, np.ndarray]:
    if not isinstance(healpix_grid, Grid):
        healpix_grid = Grid(**healpix_grid)
    if not isinstance(sampling_grid, SamplingGrid):
        sampling_grid = BoxSamplingGrid(**sampling_grid)

    if data.ndim not in [1, 2] or data.shape[-1] not in [3, 4]:
        raise ValueError(
            f"Data should be shaped like (N), (N, 3), or (N, 4). Received {data.shape}."
        )

    if interpolation not in interpolators:
        raise KeyError(
            f"Unknown interpolation method: '{interpolation}'. "
            f"Registered interpolators are {list(interpolators.keys())}"
        )
    interpolator_cls = interpolators[interpolation]

    # concrete sampling grid
    target_grid = sampling_grid.resolve(cell_ids, healpix_grid)

    # deduplication
    source_cell_ids, inverse_indices = np.unique(
        cell_ids, sorted=True, return_inverse=True
    )
    deduplicated = numpy_groupies.aggregate(inverse_indices, data, func=agg, axis=0)

    interpolator = interpolator_cls(
        source_cell_ids, healpix_grid=healpix_grid, target_grid=target_grid
    )
    interpolated = interpolator.interpolate(
        deduplicated, background_value=background_value
    )
    return target_grid, interpolated
