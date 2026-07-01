Examples
========

Synthetic Dose Example
----------------------

.. code-block:: python

   from contdid import ContDIDSpec, PanelData
   from contdid import estimate_dose_effects, simulate_contdid_data

   frame = simulate_contdid_data(
       n=500,
       num_time_periods=2,
       num_groups=2,
       seed=1234,
   )
   panel = PanelData(frame=frame)
   spec = ContDIDSpec(
       target_parameter="level",
       aggregation="dose",
       dose_est_method="parametric",
       control_group="nevertreated",
       bstrap=False,
   )

   result = estimate_dose_effects(panel, spec, degree=1, num_knots=0)
   print(result.estimand)
   print(result.grid[:3])
   print(result.estimate[:3])

Release Examples
----------------

The repository also ships checked release-example scripts:

- ``examples/plot_dose_curve.py``
- ``examples/medicare_release_walkthrough.py``

The Medicare walkthrough is an empirical scaffold unless licensed source data
are available. It must not be interpreted as licensed application parity.

Real-world Medicare scaffold walkthrough
----------------------------------------

The checked Medicare scaffold is the recommended real-world tutorial entry
point for release consumers. It mirrors the Acemoglu-Finkelstein Medicare PPS
application window from the paper, but it deliberately uses a small checked
scaffold and the public-source honesty boundary instead of claiming licensed
1980-1986 AHA parity.

The paper application uses AHA annual-survey hospital data for 1980-1986, fixes
1983 as the baseline year, and treats the 1983 Medicare inpatient share as the
continuous dose. The checked public substitute registry records
``cms_hcris_hospital_cost_reports`` as ``descriptive-or-scaffold-only`` because
public HCRIS coverage starts in 1996. Its
``parity_claim_allowed=False`` flag means the scaffold can teach the package
flow, but it is not licensed Medicare PPS parity evidence.

The checked inputs are intentionally small and inspectable:

- ``reproduction/phase9_release_examples/medicare-scaffold-demo/two_period_panel.csv``
- ``reproduction/phase9_release_examples/medicare-scaffold-demo/eventstudy_panel.csv``
- ``reproduction/phase9_release_examples/medicare-scaffold-demo/metadata.json``
- ``reproduction/medicare_pps/source_options.json``

Use the package public API to inspect the source boundary before estimating:

.. code-block:: python

   import pandas as pd

   from contdid import (
       ContDIDSpec,
       PanelData,
       estimate_eventstudy_effects,
       load_medicare_pps_source_options,
       prepare_medicare_pps_panel,
   )

   source_options = load_medicare_pps_source_options()
   hcris = next(
       source for source in source_options["public_substitutes"]
       if source["id"] == "cms_hcris_hospital_cost_reports"
   )
   assert hcris["allowed_use"] == "descriptive-or-scaffold-only"
   assert hcris["parity_claim_allowed"] is False

   scaffold = pd.read_csv(
       "reproduction/phase9_release_examples/medicare-scaffold-demo/"
       "eventstudy_panel.csv"
   )
   prepared = prepare_medicare_pps_panel(
       scaffold,
       unit_column="id",
       year_column="time_period",
       outcome_column="Y",
       dose_column="D",
       source_id="cms_hcris_hospital_cost_reports",
   )
   assert prepared.metadata["allowed_use"] == "descriptive-or-scaffold-only"

   spec = ContDIDSpec(
       target_parameter="level",
       aggregation="eventstudy",
       control_group="nevertreated",
       dose_est_method="parametric",
       bstrap=False,
   )
   result = estimate_eventstudy_effects(
       PanelData(scaffold),
       spec,
       degree=1,
       num_knots=0,
       base_period=1983,
   )

   display_table = result.to_frame()
   print(display_table)

   markdown_table = result.to_markdown(include_caption=True, digits=3)
   print(markdown_table)

For docs-ready release outputs, run:

.. code-block:: bash

   python3 examples/medicare_release_walkthrough.py

That script writes the checked Markdown and JSON consumer artifacts under
``reproduction/phase9_release_examples/consumer-outputs/`` while preserving the
``descriptive-or-scaffold-only`` label and the
``cms_hcris_hospital_cost_reports`` source boundary. Use
``ContDIDResult.to_frame()`` for notebook tables,
``ContDIDResult.to_markdown(include_caption=True, digits=3)`` for
docs-ready report tables, ``ContDIDResult.save_plot()`` for a
checked PNG that draws the same result axis, intervals, bands, zero lines, and
event-study support markers, and keep
``estimate_eventstudy_effects`` on the checked event-study route; do not relabel
the scaffold as licensed Medicare PPS parity unless the licensed AHA 1980-1986 inputs are
available and the parity contract is updated.

The checked walkthrough JSON also keeps the release route inspectable for
tutorial readers: ``source_surface`` must stay ``prepare_medicare_pps_panel``, while
``package_surfaces`` maps the public estimators used by the release packet
(``estimate_dose_effects``, ``estimate_dose_slope_effects``, and
``estimate_eventstudy_effects``). Treat those fields as the tutorial provenance
check before reusing the scaffold in another notebook or documentation page.
