API reference
=============

Main function
-------------

.. autosummary::
   :toctree: generated

   healpix_plot.plot

Low-level function
------------------

`resample` is called internally by `plot`. Use it directly only if you need the raw image array without rendering (e.g. to post-process it or display it with a different tool).

.. autosummary::
   :toctree: generated

   healpix_plot.resample

Classes
-------

.. autosummary::
   :toctree: generated

   healpix_plot.HealpixGrid
   healpix_plot.HEALPix
   healpix_plot.sampling_grid.ParametrizedSamplingGrid
   healpix_plot.sampling_grid.AffineSamplingGrid
   healpix_plot.sampling_grid.ConcreteSamplingGrid

Projection and rasters
----------------------

Helpers for PROJ's ellipsoidal HEALPix projection, and for exchanging
resampled images with georeferenced rasters (GeoTIFF via rioxarray).

.. autosummary::
   :toctree: generated

   healpix_plot.projection.proj_string
   healpix_plot.projection.healpix_crs
   healpix_plot.projection.geographic_crs
   healpix_plot.projection.cell_aligned_grid
   healpix_plot.raster.to_dataarray
   healpix_plot.raster.raster_cell_ids
