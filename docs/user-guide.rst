User Guide
==========

Installation
------------

Install the package from the repository root with an editable install:

.. code-block:: bash

   pip install -e contdid-py

Core Objects
------------

``PanelData`` wraps a balanced long panel and records the column names used by
the runtime validators and estimators. The default columns are ``id``,
``time_period``, ``Y``, ``G``, and ``D``.

``ContDIDSpec`` records the requested estimand, aggregation, dose estimator,
control group, and inference controls. The checked public routes currently use
continuous treatments and hard-fail unsupported treatment types or unsupported
CCK/event-study combinations.

``ContDIDResult`` stores the public result payload: estimand label, grid,
estimate, standard error, optional critical value, confidence interval,
confidence band, and metadata.

Display Tables and Plots
------------------------

Use ``ContDIDResult.to_frame()`` when a notebook or downstream script needs a
typed ``pandas.DataFrame``. Use ``ContDIDResult.to_markdown()`` when a report,
README, or release packet needs a compact table that can be pasted directly into
Markdown. Pass boolean ``include_caption=True`` when the table should carry its own
estimand, row count, axis, and critical value above the Markdown grid without
printing the full metadata dictionary. Pass ``digits=3`` or another integer from
0 through 12 when a manuscript table needs display-only rounding while the result
object keeps the full stored estimates, standard errors, intervals, and metadata.

The Markdown table keeps the public display columns stable:

.. code-block:: markdown

   | Event time | Estimate | Std. error | Pointwise CI | Uniform band | Support |
   | ---: | ---: | ---: | --- | --- | --- |
   | -1 | -0.100000 | 0.200000 | [-0.500000, 0.300000] | not estimated (uniform band) | yes |
   | 0 | 0.200000 | 0.100000 | [0.000000, 0.400000] | not estimated (uniform band) | yes |
   | 1 | 0.500000 | 0.300000 | [0.100000, 0.900000] | not estimated (uniform band) | no |

The numeric cells use fixed six-decimal formatting by default, confidence
intervals and uniform confidence bands are bracketed when present, and event-study
support is rendered with yes/no event-study support labels.

Use ``ContDIDResult.save_plot()`` when a report or notebook needs a
publication-style PNG directly from the checked result object. The plot uses the
same ``dose`` or ``event_time`` axis as ``to_frame()``, renders exported
pointwise confidence intervals and uniform confidence bands when available,
marks the zero reference line, and shows event-study support diagnostics when
the result carries support metadata. The method writes only PNG output and
returns the saved ``pathlib.Path``.

Real-World Tutorial Provenance
------------------------------

The checked Medicare scaffold tutorial is descriptive-or-scaffold-only. Before
reusing it in a notebook, inspect
``reproduction/phase9_release_examples/consumer-outputs/medicare_release_walkthrough.json``:
``source_surface`` must stay ``prepare_medicare_pps_panel`` and
``package_surfaces`` must point to the checked public estimators used by the
packet. Those fields keep the source route separate from the estimator routes,
so the tutorial cannot be mistaken for licensed Medicare PPS parity evidence.

Supported Public Routes
-----------------------

The checked v1 Python surface exposes:

- ``simulate_contdid_data`` for synthetic panels.
- ``estimate_dose_effects`` and ``estimate_dose_level_effects`` for ``ATT(d)``.
- ``estimate_dose_slope_effects`` for ``ACRT(d)``.
- ``estimate_eventstudy_effects`` for ``ATT(event_time)``.
- ``estimate_eventstudy_slope_effects`` for ``ACRT(event_time)``.
- ``build_confidence_band`` and ``compute_multiplier_bootstrap`` for inference
  payload construction.

CCK Boundary
------------

The checked CCK dose route is deliberately narrow. It is only supported for
``aggregation="dose"`` with two observed time periods, one positive treatment-timing cohort,
positive treatment timing to start in the post period, and an untreated ``D == 0``
benchmark. Requests outside that shape hard-fail instead of falling through to
an unchecked approximation.

The public error surface is part of the release contract. Staggered-adoption CCK
requests raise ``cck estimator not supported with staggered adoption yet`` before
the generic multi-period or event-study guards. CCK event-study requests raise
``event study not supported with cck estimator yet``; ``base_period`` and
``control_group`` options must not relax that boundary. The checked event-study
control groups remain ``notyettreated`` and ``nevertreated`` for the parametric
event-study routes.

The runtime CCK backend is a fixed quadratic polynomial scaffold used for the
supported two-period dose surface. It does not implement the paper's
data-driven K-hat, Lepski, or ``npiv`` sieve selection, so release-facing claims
must not describe it as full adaptive CCK parity. Event-study inference also
requires locally identified post-treatment support with inference degrees of freedom
before reporting uncertainty.

Data Rules
----------

Real-world datasets used for audits, cross-checks, or regression tests must be
placed under the repository-level ``data/`` directory with source, license, and
consumer notes. Synthetic fixtures and generated Monte Carlo outputs may remain
with their test or reproduction bundles.
