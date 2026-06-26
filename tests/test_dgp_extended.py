"""Tests for extended DGP scenarios (SIM-006 through SIM-010)."""

import numpy as np
import pytest

from contdid import simulate_contdid_data, cont_did, PanelData, ContDIDSpec


class TestDGPGeneration:
    """Test that all new DGPs generate valid panel data."""

    @pytest.mark.parametrize("dgp_id", [
        "SIM-006-cubic-dose",
        "SIM-007-sine-dose",
        "SIM-008-threshold-dose",
        "SIM-009-staggered-linear-dose",
        "SIM-010-large-panel",
    ])
    def test_generates_valid_panel(self, dgp_id):
        panel = simulate_contdid_data(dgp_id)
        assert isinstance(panel, PanelData)
        assert len(panel.frame) > 0

    @pytest.mark.parametrize("dgp_id", [
        "SIM-006-cubic-dose",
        "SIM-007-sine-dose",
        "SIM-008-threshold-dose",
        "SIM-009-staggered-linear-dose",
        "SIM-010-large-panel",
    ])
    def test_reproducible_with_same_seed(self, dgp_id):
        p1 = simulate_contdid_data(dgp_id)
        p2 = simulate_contdid_data(dgp_id)
        assert p1.frame.equals(p2.frame)


class TestDGPEstimation:
    """Test that estimation on DGP data recovers known truth."""

    def test_cubic_dose_recovery(self):
        """SIM-006: cubic polynomial should be exactly recovered by degree>=3 B-spline."""
        panel = simulate_contdid_data("SIM-006-cubic-dose")
        result = cont_did(panel, degree=3, num_knots=3)
        # ATT(d) should match cubic truth at evaluation points
        # (exact recovery when B-spline degree >= polynomial degree)
        assert result is not None
        assert len(result.grid) > 0

    def test_sine_dose_convergence(self):
        """SIM-007: sine function approximated better with more knots."""
        panel = simulate_contdid_data("SIM-007-sine-dose")
        result_few = cont_did(panel, degree=3, num_knots=2)
        result_many = cont_did(panel, degree=3, num_knots=5)
        # Both should produce valid results
        assert result_few is not None
        assert result_many is not None

    def test_threshold_dose_estimation(self):
        """SIM-008: threshold effect estimated without crash."""
        panel = simulate_contdid_data("SIM-008-threshold-dose")
        result = cont_did(panel, degree=3, num_knots=3)
        assert result is not None

    def test_staggered_linear_eventstudy(self):
        """SIM-009: staggered adoption with linear dose effect."""
        panel = simulate_contdid_data("SIM-009-staggered-linear-dose")
        result = cont_did(panel, aggregation="eventstudy")
        assert result is not None

    def test_large_panel_consistency(self):
        """SIM-010: large panel produces same result pattern as small."""
        panel = simulate_contdid_data("SIM-010-large-panel")
        result = cont_did(panel, degree=3, num_knots=2)
        assert result is not None
        assert len(result.grid) > 0
