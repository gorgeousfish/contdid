# contdid

**Continuous Treatment Difference-in-Differences**

Estimate the full dose-response curve when treatments vary in intensity, rather than reducing continuous policy variation to a single coefficient.

## Statement of Need

Standard Difference-in-Differences (DiD) tools estimate a single average treatment effect for a binary intervention. Yet many policies of interest are continuous: tax rate changes differ across regions, subsidy amounts vary by firm size, pollution exposure levels differ by proximity to a source, and healthcare reimbursement cuts vary by hospital.

Collapsing continuous variation into binary "treated vs. untreated" discards the dose-response structure that is often central to policy evaluation. Researchers need to know *how much* the outcome responds as dose increases, not just *whether* it responds.

**contdid** provides a Python implementation of the Caetano, Callaway, Payne & Rodrigues (2024) continuous-treatment DiD framework. It estimates ATT(*d*), the average treatment effect on the treated as a function of dose *d*, recovering the shape of how outcomes respond to treatment intensity. No other Python package currently implements these estimators.

## Key Concepts

**ATT(*d*)**: Average Treatment Effect on the Treated at dose *d*.

$$
\text{ATT}(d) = E[Y_t(d) - Y_t(0) \mid D = d, G \leq t]
$$

The effect of receiving dose *d* compared to receiving no treatment, for units who actually received dose *d*.

**ACRT(*d*)**: Average Causal Response Trajectory.

$$
\text{ACRT}(d) = \frac{\partial}{\partial d} \text{ATT}(d)
$$

The marginal effect of an additional unit of dose at level *d*.

**Identification assumption**: Conditional parallel trends: absent treatment, units at every dose level would have followed parallel outcome paths to untreated units.

## When to Use / When Not to Use

| Good fit                                        | Not designed for                                  |
| ----------------------------------------------- | ------------------------------------------------- |
| Panel data with continuous treatment intensity  | Pure cross-sectional data (need panel structure)  |
| Staggered treatment adoption with varying doses | Binary treatment only (use standard DiD packages) |
| Interest in the full dose-response curve shape  | No untreated comparison group available           |
| Need for uniform inference (simultaneous bands) | Instrumental variable identification              |
| Multi-period event-study with continuous dose   | Continuous-time survival/duration models          |

## Installation

```bash
pip install git+https://github.com/gorgeousfish/contdid.git
```

With plotting support:

```bash
pip install "contdid[plotting] @ git+https://github.com/gorgeousfish/contdid.git"
```

**Requirements:** Python ≥ 3.11, numpy, pandas, scipy

## Quick Start

```python
from contdid import cont_did, simulate_contdid_data, summary

# Simulate a two-period panel with known linear+quadratic dose effect
panel = simulate_contdid_data(n=2000, dgp_id="SIM-005-cck-two-period", seed=42)

# One-line estimation with bootstrap inference
result = cont_did(panel)
print(summary(result, max_rows=10))
```

Output:

```
========================================================================
                       ContDID Estimation Results                     
========================================================================
Estimand:           ATT(d)
Inference:          bootstrap
Confidence level:   95%
Basis:              global_polynomial (degree=3, knots=0)
------------------------------------------------------------------------
      Grid    Estimate    Std.Err.    CI Lower    CI Upper
----------------------------------------------------------
    0.0889      0.1811      0.0992     -0.0133      0.3756
    0.1904      0.2919      0.0872      0.1211      0.4628
    0.2836      0.3873      0.0900      0.2109      0.5637
    0.3944      0.4963      0.0847      0.3303      0.6623
    0.4890      0.5883      0.0800      0.4315      0.7451
    0.5719      0.6703      0.0819      0.5098      0.8308
    0.6780      0.7798      0.0880      0.6072      0.9524
    0.7875      0.9017      0.0873      0.7306      1.0728
    0.8866      1.0230      0.0923      0.8421      1.2040
    0.9921      1.1670      0.1634      0.8468      1.4872
  ... (10 of 90 points shown, equidistant sampling)
------------------------------------------------------------------------
Critical value:     1.9600
Band type:          pointwise_multiplier
========================================================================
```

### Scenario 1: Event-Study Design

Track how effects evolve over time since treatment onset:

```python
from contdid import cont_did, simulate_contdid_data, summary

panel = simulate_contdid_data(n=2000, dgp_id="SIM-004-staggered-eventstudy-null", seed=42)
result_es = cont_did(panel, aggregation="eventstudy")
print(summary(result_es))
```

Output:

```
========================================================================
                       ContDID Estimation Results                     
========================================================================
Estimand:           ATT(event_time)
Inference:          bootstrap
Confidence level:   95%
Basis:              global_polynomial (degree=3, knots=0)
------------------------------------------------------------------------
      Grid    Estimate    Std.Err.    CI Lower    CI Upper
----------------------------------------------------------
   -2.0000      0.0762      0.0864     -0.0930      0.2455
   -1.0000      0.0339      0.0564     -0.0767      0.1445
    0.0000      0.0700      0.0454     -0.0189      0.1589
    1.0000      0.0489      0.0617     -0.0721      0.1698
    2.0000      0.0361      0.0877     -0.1359      0.2080
------------------------------------------------------------------------
Critical value:     1.9600
Band type:          pointwise_multiplier
========================================================================
```

### Scenario 2: CCK Nonparametric with Uniform Bands

Use sieve estimation with simultaneous confidence bands for shape inference:

```python
from contdid import cont_did, simulate_contdid_data, summary

panel = simulate_contdid_data(n=2000, dgp_id="SIM-005-cck-two-period", seed=42)
result_cck = cont_did(panel, dose_est_method="cck", cband=True)
print(summary(result_cck, max_rows=8))
```

Output:

```
========================================================================
                       ContDID Estimation Results                     
========================================================================
Estimand:           ATT(d)
Inference:          bootstrap
Confidence level:   95%
Basis:              cck_polynomial_backend (degree=2, knots=0)
------------------------------------------------------------------------
      Grid    Estimate    Std.Err.    CI Lower    CI Upper
----------------------------------------------------------
    0.0010      0.1005      0.1303     -0.1550      0.3559
    0.1437      0.2375      0.0851      0.0707      0.4042
    0.2864      0.3790      0.0750      0.2320      0.5259
    0.4290      0.5249      0.0791      0.3700      0.6799
    0.5717      0.6754      0.0788      0.5209      0.8300
    0.7144      0.8304      0.0747      0.6841      0.9768
    0.8571      0.9899      0.0858      0.8218      1.1580
    0.9998      1.1539      0.1325      0.8941      1.4137
  ... (8 of 50 points shown, equidistant sampling)
------------------------------------------------------------------------
Critical value:     2.6385
Band type:          simultaneous_multiplier
========================================================================
```

### Scenario 3: Pre-Trend Testing

Diagnose whether parallel trends holds in pre-treatment periods:

```python
from contdid import cont_did, simulate_contdid_data, pre_trend_test_from_result

panel = simulate_contdid_data(n=2000, dgp_id="SIM-004-staggered-eventstudy-null", seed=42)
result_es = cont_did(panel, aggregation="eventstudy")

ptr = pre_trend_test_from_result(result_es)
print(f"Wald statistic: {ptr.test_statistic:.4f}")
print(f"p-value:        {ptr.p_value:.4f}")
print(f"DoF:            {ptr.degrees_of_freedom}")
print(f"Reject at 5%:   {ptr.reject_at_05}")
```

Output:

```
Wald statistic: 1.2942
p-value:        0.5236
DoF:            2
Reject at 5%:   False
```

## Method Selection Guide

| Parameter            | Option              | Description                                                                                                      |
| -------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `dose_est_method`  | `"parametric"`    | B-spline OLS; flexible, supports multi-period                                                                   |
|                      | `"cck"`           | CCK sieve estimation; nonparametric, two-period only                                                            |
| `aggregation`      | `"dose"`          | Dose-response curve ATT(*d*)                                                                                   |
|                      | `"eventstudy"`    | Event-time dynamics ATT(*event_time*)                                                                          |
| `control_group`    | `"nevertreated"`  | Never-treated units as comparison                                                                                |
|                      | `"notyettreated"` | Not-yet-treated units (staggered designs)                                                                        |
| `target_parameter` | `"level"`         | ATT (level effects)                                                                                              |
|                      | `"slope"`         | ACRT (marginal/derivative effects)                                                                               |
| `cband`            | `False`           | Pointwise confidence intervals                                                                                   |
|                      | `True`            | Simultaneous confidence band (uniform inference)                                                                 |
| `adaptive`         | `True`            | Lepski adaptive dimension selection (CCK only)                                                                   |
| `knot_method`      | `"quantile"`      | `"quantile"` or `"even"`. Quantile-based adapts to dose distribution; even-spaced provides uniform coverage. |

## API Overview

### Core Workflow

```python
from contdid import (
    simulate_contdid_data,  # 1. Generate/load panel data
    cont_did,               # 2. One-line estimation (recommended)
    summary,                # 3. View formatted results
    to_dataframe,           # 4. Export to DataFrame
    to_csv,                 #    Export to CSV
    to_latex,               #    Export to LaTeX
    plot_dose_response,     # 5. Visualize dose-response
    plot_eventstudy,        #    Visualize event-study
    pre_trend_test_from_result,  # 6. Diagnose parallel trends
    quantile_knots,         # 7. Knot placement utilities
    even_knots,
)
```

### `cont_did()` Parameters

```python
result = cont_did(
    panel,                          # PanelData object
    target_parameter="level",       # "level" (ATT) or "slope" (ACRT)
    aggregation="dose",             # "dose" or "eventstudy"
    dose_est_method="parametric",   # "parametric" or "cck"
    control_group="nevertreated",   # "nevertreated" or "notyettreated"
    degree=3,                       # B-spline polynomial degree
    num_knots=0,                    # Interior knots (0 = global polynomial)
    knot_method="quantile",          # "quantile" or "even" knot placement
    anticipation=0,                 # Anticipation periods
    biters=1000,                    # Bootstrap iterations
    cband=False,                    # Simultaneous confidence band
    adaptive=False,                 # Lepski adaptive (CCK two-period only)
)
```

### `ContDIDResult` Object

| Attribute / Method             | Description                                |
| ------------------------------ | ------------------------------------------ |
| `result.grid`                | Evaluation points (dose or event-time)     |
| `result.estimate`            | Point estimates at each grid point         |
| `result.std_error`           | Bootstrap standard errors                  |
| `result.metadata`            | Dict with CI bounds, band type, basis info |
| `summary(result)`            | Formatted text table                       |
| `to_dataframe(result)`       | pandas DataFrame export                    |
| `to_csv(result, path)`       | CSV file export                            |
| `to_latex(result)`           | LaTeX table string                         |
| `plot_dose_response(result)` | Dose-response figure                       |
| `plot_eventstudy(result)`    | Event-study figure                         |

## Performance

Bootstrap inference uses thread-level parallelism (`ThreadPoolExecutor`) for
large-scale problems (biters >= 200, multiple chunks). NumPy releases the GIL
during matrix operations, enabling 15-67% speedup on multi-core machines.
Results are reproducible via `numpy.random.SeedSequence`: the same seed
always yields identical output regardless of thread count.

## Current Limitations & Future Directions

### Current Limitations

- CCK estimation restricted to two-period, non-staggered designs
- Lepski adaptive dimension selection restricted to two-period panels
- Event-study CCK supports fixed-dimension mode only (no adaptive Lepski)
- Large panels (N > 50,000) benefit from the built-in thread-parallel bootstrap, though wall time scales linearly with `biters`
- **Covariate adjustment**: Not available. The paper (arXiv:2107.02637v7) provides only a conceptual framework for conditional parallel trends without complete estimation theory.
- **Discrete treatment**: Not available. Only continuous treatment is implemented; multi-valued discrete treatment (paper Assumption 4b) awaits implementation.

### Future Directions

- Covariate adjustment under conditional parallel trends, pending complete estimation theory for the propensity-score reweighting step
- Multi-valued discrete treatment via the saturated regression estimator of Assumption 4b in Caetano et al. (2024)
- Joint coverage theory for multi-period CCK aggregation (extending the two-period uniform band to event-study designs)
- Additional DGP scenarios for simulation and testing

## Citation

**APA:**

Cai, X., & Xu, W. (2025). *contdid-py: Continuous Treatment Difference-in-Differences for Python* (Version 0.1.0) [Computer software]. https://github.com/gorgeousfish/contdid

**BibTeX:**

```bibtex
@software{contdid_py,
  title = {contdid-py: Continuous Treatment Difference-in-Differences for Python},
  author = {Cai, Xuanyu and Xu, Wenli},
  year = {2025},
  url = {https://github.com/gorgeousfish/contdid},
  version = {0.1.0}
}
```

**Method papers:**

```bibtex
@article{caetano2024continuous,
  title = {Difference in Differences with Continuous Treatment},
  author = {Caetano, Gregorio and Callaway, Brantly and Payne, Stroud and Rodrigues, Hugo Sant'Anna},
  year = {2024},
  journal = {arXiv preprint arXiv:2107.02637v7},
  url = {https://arxiv.org/abs/2107.02637}
}

@article{chen2024adaptive,
  title = {Adaptive Estimation and Uniform Confidence Bands for Nonparametric Structural Functions and Elasticities},
  author = {Chen, Xiaohong and Christensen, Timothy M. and Kankanala, Siddhartha},
  year = {2024},
  journal = {arXiv preprint arXiv:2107.11869v3},
  url = {https://arxiv.org/abs/2107.11869}
}
```

## Authors

**Python Implementation:**

- Xuanyu Cai, City University of Macau, xuanyuCAI@outlook.com
- Wenli Xu, City University of Macau, wlxu@cityu.edu.mo

**Methodology:**

- Gregorio Caetano, University of Georgia
- Brantly Callaway, University of Georgia
- Tymon Słoczyński, Brandeis University

Based on:

- Caetano, G., Callaway, B., Payne, S., & Rodrigues, H. S. (2024). "Difference in Differences with Continuous Treatment." arXiv:2107.02637v7.
- Chen, X., Christensen, T. M., & Kankanala, S. (2024). "Adaptive Estimation and Uniform Confidence Bands for Nonparametric Structural Functions and Elasticities." arXiv:2107.11869v3.

## Related Packages

| Package                                        | Language | Description                                 |
| ---------------------------------------------- | -------- | ------------------------------------------- |
| [contdid](https://github.com/bcallaway11/contdid) | R        | Reference implementation by Callaway et al. |
| [contdid-stata](https://github.com/contdid/contdid-stata) | Stata    | Stata implementation for continuous DiD     |
| [did](https://github.com/bcallaway11/did)         | R        | Binary treatment staggered DiD              |
| [pydid](https://github.com/d2cml-ai/pydid)        | Python   | Binary treatment DiD for Python             |

## License

[AGPL-3.0](LICENSE)
