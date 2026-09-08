API reference
=============

Main function
-------------

.. autosummary::
   :toctree: generated

   healpix_plot.plot
   healpix_plot.pcolormesh

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
   healpix_plot.sampling_grid.ParametrizedSamplingGrid
   healpix_plot.sampling_grid.AffineSamplingGrid
   healpix_plot.sampling_grid.ConcreteSamplingGrid
