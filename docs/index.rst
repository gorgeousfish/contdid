contdid documentation
=====================

``contdid`` is the checked Python release surface for
continuous-treatment Difference-in-Differences estimators. The package keeps
the public API traceable to the paper contracts, the reference R package, and
the release verification artifacts in this repository.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   user-guide
   examples
   api

Status
------

The current Python package is an alpha release. The v1 surface includes checked
input validation, synthetic data generation, parametric dose estimators,
event-study routes, inference payload helpers, empirical scaffold metadata, and
release-facing examples. Unsupported identification surfaces hard-fail rather
than silently changing estimands.
