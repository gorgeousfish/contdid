"""contdid: Continuous Treatment Difference-in-Differences for Python."""

__version__ = "0.1.0"

from .api import cont_did, pre_trend_test_from_result
from .bspline import even_knots, quantile_knots
from .data import PanelData
from .empirical import (
    load_medicare_pps_source_options,
    prepare_medicare_pps_panel,
    resolve_medicare_pps_source,
)
from .estimation import (
    build_dose_grid,
    estimate_dose_effects,
    estimate_dose_level_effects,
    estimate_dose_slope_effects,
)
from .eventstudy import estimate_eventstudy_effects, estimate_eventstudy_slope_effects
from .inference import (
    attach_inference_payload,
    build_confidence_band,
    compute_multiplier_bootstrap,
)
from .plotting import plot_dose_response, plot_eventstudy
from .results import ContDIDResult
from .simulate import simulate_contdid_data
from .specs import ContDIDSpec
from .summary import summary, to_csv, to_dataframe, to_latex
from .validation import (
    ContDIDValidationError,
    validate_panel_data,
    validate_spec,
)

__all__ = [
    "__version__",
    # High-level API
    "cont_did",
    "pre_trend_test_from_result",
    # Core data structures
    "PanelData",
    "ContDIDSpec",
    "ContDIDResult",
    # Estimation
    "build_dose_grid",
    "estimate_dose_effects",
    "estimate_dose_level_effects",
    "estimate_dose_slope_effects",
    "estimate_eventstudy_effects",
    "estimate_eventstudy_slope_effects",
    # Inference
    "attach_inference_payload",
    "build_confidence_band",
    "compute_multiplier_bootstrap",
    # Simulation
    "simulate_contdid_data",
    # Output
    "summary",
    "to_dataframe",
    "to_csv",
    "to_latex",
    "plot_dose_response",
    "plot_eventstudy",
    # Utilities
    "quantile_knots",
    "even_knots",
    # Data helpers
    "load_medicare_pps_source_options",
    "prepare_medicare_pps_panel",
    "resolve_medicare_pps_source",
    # Validation
    "ContDIDValidationError",
    "validate_panel_data",
    "validate_spec",
]
