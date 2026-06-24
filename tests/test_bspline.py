"""Tests for B-spline basis construction."""
import numpy as np
import pytest

from contdid.bspline import (
    build_bspline_design,
    build_bspline_derivative_design,
    quantile_knots,
    even_knots,
)


class TestBSplineDesign:
    def test_shape_no_knots(self):
        """degree=3, no knots -> 4 basis functions"""
        dose = np.linspace(0.1, 1.0, 50)
        design = build_bspline_design(dose, degree=3, interior_knots=[])
        assert design.shape == (50, 4)

    def test_shape_with_knots(self):
        """degree=3, 2 knots -> 6 basis functions"""
        dose = np.linspace(0.1, 1.0, 50)
        design = build_bspline_design(dose, degree=3, interior_knots=[0.4, 0.7])
        assert design.shape == (50, 6)

    def test_partition_of_unity(self):
        """B-spline basis functions sum to 1 at every point"""
        dose = np.linspace(0.1, 1.0, 100)
        design = build_bspline_design(dose, degree=3, interior_knots=[0.3, 0.5, 0.7])
        np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-12)

    def test_nonnegative(self):
        """All B-spline values are nonnegative"""
        dose = np.linspace(0.1, 1.0, 100)
        design = build_bspline_design(dose, degree=3, interior_knots=[0.3, 0.5, 0.7])
        assert np.all(design >= -1e-15)

    def test_local_support(self):
        """Each B-spline has local (compact) support"""
        dose = np.linspace(0.1, 1.0, 200)
        design = build_bspline_design(dose, degree=3, interior_knots=[0.3, 0.5, 0.7])
        # Interior basis functions should have zeros in some regions
        for j in range(1, design.shape[1] - 1):
            assert np.any(design[:, j] == 0.0) or np.any(
                np.isclose(design[:, j], 0.0, atol=1e-14)
            )

    def test_polynomial_space_equivalence_no_knots(self):
        """With no knots, B-spline spans same space as polynomial of same degree.

        Regression on B-spline basis should give same fitted values as polynomial regression.
        """
        rng = np.random.default_rng(42)
        dose = rng.uniform(0.1, 1.0, 100)
        y = 1.0 + 2.0 * dose + 0.5 * dose**2 - 0.3 * dose**3 + rng.normal(0, 0.1, 100)

        # B-spline fit
        bs_design = build_bspline_design(dose, degree=3, interior_knots=[])
        bs_coef, *_ = np.linalg.lstsq(bs_design, y, rcond=None)
        bs_fitted = bs_design @ bs_coef

        # Polynomial fit (same space)
        poly_design = np.column_stack(
            [np.ones_like(dose), dose, dose**2, dose**3]
        )
        poly_coef, *_ = np.linalg.lstsq(poly_design, y, rcond=None)
        poly_fitted = poly_design @ poly_coef

        np.testing.assert_allclose(bs_fitted, poly_fitted, atol=1e-10)

    def test_degree_1_no_knots(self):
        """degree=1, no knots -> 2 basis functions (linear)"""
        dose = np.linspace(0.1, 1.0, 30)
        design = build_bspline_design(dose, degree=1, interior_knots=[])
        assert design.shape == (30, 2)
        np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-12)

    def test_degree_2_with_knots(self):
        """degree=2, 3 knots -> 6 basis functions"""
        dose = np.linspace(0.1, 1.0, 80)
        design = build_bspline_design(dose, degree=2, interior_knots=[0.3, 0.5, 0.8])
        assert design.shape == (80, 6)
        np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-12)

    def test_derivative_design_shape(self):
        dose = np.linspace(0.1, 1.0, 50)
        deriv = build_bspline_derivative_design(dose, degree=3, interior_knots=[0.4, 0.7])
        assert deriv.shape == (50, 6)

    def test_derivative_numerical_consistency(self):
        """Derivative from basis should match numerical differentiation."""
        # Use interior points to avoid boundary effects in numerical differentiation
        dose = np.linspace(0.25, 0.85, 50)
        knots = [0.4, 0.6]
        # Fix boundaries so shifted dose arrays use the same knot vector
        xmin, xmax = 0.2, 0.9
        design = build_bspline_design(dose, degree=3, interior_knots=knots, xmin=xmin, xmax=xmax)
        deriv = build_bspline_derivative_design(dose, degree=3, interior_knots=knots, xmin=xmin, xmax=xmax)

        # Use random coefficients
        rng = np.random.default_rng(123)
        coef = rng.normal(size=design.shape[1])

        # Analytic derivative
        analytic = deriv @ coef

        # Numerical derivative
        h = 1e-7
        dose_plus = dose + h
        dose_minus = dose - h
        design_plus = build_bspline_design(dose_plus, degree=3, interior_knots=knots, xmin=xmin, xmax=xmax)
        design_minus = build_bspline_design(dose_minus, degree=3, interior_knots=knots, xmin=xmin, xmax=xmax)
        numerical = ((design_plus @ coef) - (design_minus @ coef)) / (2 * h)

        np.testing.assert_allclose(analytic, numerical, atol=1e-5)

    def test_condition_number_better_than_truncated_power(self):
        """B-spline design should have better conditioning than truncated power."""
        dose = np.linspace(0.1, 1.0, 200)
        knots = [0.25, 0.5, 0.75]

        bs_design = build_bspline_design(dose, degree=3, interior_knots=knots)

        # Truncated power for comparison
        columns = [np.ones_like(dose), dose, dose**2, dose**3]
        for k in knots:
            columns.append(np.clip(dose - k, 0, None) ** 3)
        tp_design = np.column_stack(columns)

        bs_cond = np.linalg.cond(bs_design)
        tp_cond = np.linalg.cond(tp_design)

        assert bs_cond < tp_cond


class TestDerivativeDesign:
    def test_derivative_sums_to_zero(self):
        """Derivatives of B-spline basis sum to 0 (derivative of partition of unity)."""
        dose = np.linspace(0.2, 0.9, 50)
        deriv = build_bspline_derivative_design(
            dose, degree=3, interior_knots=[0.3, 0.5, 0.7]
        )
        np.testing.assert_allclose(deriv.sum(axis=1), 0.0, atol=1e-10)

    def test_linear_derivative(self):
        """For degree=1, derivative should be piecewise constant."""
        dose = np.linspace(0.1, 1.0, 100)
        deriv = build_bspline_derivative_design(dose, degree=1, interior_knots=[0.5])
        # With degree 1 and 1 knot: 3 basis functions, derivative is piecewise constant
        assert deriv.shape == (100, 3)


class TestKnotSelection:
    def test_quantile_knots_count(self):
        dose = np.linspace(0.1, 1.0, 100)
        knots = quantile_knots(dose, 3)
        assert len(knots) == 3

    def test_quantile_knots_ordered(self):
        dose = np.random.default_rng(42).uniform(0.1, 1.0, 100)
        knots = quantile_knots(dose, 5)
        assert all(knots[i] < knots[i + 1] for i in range(len(knots) - 1))

    def test_quantile_knots_within_range(self):
        dose = np.random.default_rng(42).uniform(0.1, 1.0, 100)
        knots = quantile_knots(dose, 3)
        assert all(dose.min() < k < dose.max() for k in knots)

    def test_even_knots_count(self):
        dose = np.linspace(0.1, 1.0, 100)
        knots = even_knots(dose, 4)
        assert len(knots) == 4

    def test_even_knots_spacing(self):
        dose = np.linspace(0.0, 1.0, 100)
        knots = even_knots(dose, 3)
        diffs = np.diff(knots)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-12)

    def test_zero_knots(self):
        dose = np.linspace(0.1, 1.0, 100)
        assert quantile_knots(dose, 0) == []
        assert even_knots(dose, 0) == []

    def test_even_knots_within_range(self):
        dose = np.linspace(0.2, 0.8, 100)
        knots = even_knots(dose, 4)
        assert all(dose.min() < k < dose.max() for k in knots)
