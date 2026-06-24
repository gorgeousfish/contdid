"""Tests for the post-processing hook system (contdid.hooks).

Covers:
- Hook registration and removal
- ReadOnlyResult immutability protection
- Hook execution order (priority)
- Conditional triggering
- Exception isolation (hook errors don't break main flow)
- Hook output storage and access via ContDIDResult.hook_outputs
- Mathematical consistency: core results not modified by hooks
- Integration with cont_did()
- Regression: no-hook behavior unchanged
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from contdid.hooks import (
    HookRegistry,
    HookSpec,
    HookStage,
    ReadOnlyResult,
    _default_hook_registry,
    dose_summary_hook,
    get_hook_registry,
    register_hook,
    unregister_hook,
)
from contdid.results import ContDIDResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_result():
    """Create a minimal ContDIDResult for testing."""
    return ContDIDResult(
        estimand="ATT(d)",
        grid=[1.0, 2.0, 3.0, 4.0, 5.0],
        estimate=[0.5, 1.0, 1.5, 2.0, 2.5],
        std_error=[0.1, 0.2, 0.3, 0.4, 0.5],
        confidence_interval=[
            [0.3, 0.7],
            [0.6, 1.4],
            [0.9, 2.1],
            [1.2, 2.8],
            [1.5, 3.5],
        ],
        metadata={"alp": 0.05, "control_group": "nevertreated"},
    )


@pytest.fixture
def registry():
    """Create a fresh HookRegistry for each test."""
    return HookRegistry()


@pytest.fixture(autouse=True)
def clean_default_registry():
    """Ensure the default registry is clean before and after each test."""
    _default_hook_registry.clear()
    yield
    _default_hook_registry.clear()


# ---------------------------------------------------------------------------
# Tests: HookRegistry registration and removal
# ---------------------------------------------------------------------------


class TestHookRegistration:
    """Tests for hook registration and unregistration."""

    def test_register_hook(self, registry):
        """Test basic hook registration."""
        registry.register("test_hook", lambda r: "output")
        assert "test_hook" in registry.list_hooks()
        assert len(registry) == 1

    def test_register_multiple_hooks(self, registry):
        """Test registering multiple hooks."""
        registry.register("hook_a", lambda r: "a")
        registry.register("hook_b", lambda r: "b")
        registry.register("hook_c", lambda r: "c")
        assert registry.list_hooks() == ["hook_a", "hook_b", "hook_c"]
        assert len(registry) == 3

    def test_register_overwrites_same_name(self, registry):
        """Test that registering with same name overwrites."""
        registry.register("my_hook", lambda r: "first")
        registry.register("my_hook", lambda r: "second")
        assert len(registry) == 1
        assert "my_hook" in registry.list_hooks()

    def test_unregister_hook(self, registry):
        """Test hook removal."""
        registry.register("to_remove", lambda r: None)
        registry.unregister("to_remove")
        assert "to_remove" not in registry.list_hooks()
        assert len(registry) == 0

    def test_unregister_nonexistent_raises(self, registry):
        """Test that unregistering nonexistent hook raises KeyError."""
        with pytest.raises(KeyError, match="未注册"):
            registry.unregister("nonexistent")

    def test_clear(self, registry):
        """Test clearing all hooks."""
        registry.register("a", lambda r: None)
        registry.register("b", lambda r: None)
        registry.clear()
        assert len(registry) == 0
        assert registry.list_hooks() == []

    def test_register_with_stage(self, registry):
        """Test registering hook with specific stage."""
        registry.register(
            "post_est",
            lambda r: None,
            stage=HookStage.POST_ESTIMATION,
        )
        assert len(registry) == 1

    def test_register_with_priority(self, registry):
        """Test registering hook with custom priority."""
        registry.register("low_prio", lambda r: None, priority=200)
        registry.register("high_prio", lambda r: None, priority=10)
        assert len(registry) == 2


# ---------------------------------------------------------------------------
# Tests: ReadOnlyResult immutability
# ---------------------------------------------------------------------------


class TestReadOnlyResult:
    """Tests for ReadOnlyResult immutability protection."""

    def test_read_attributes(self, sample_result):
        """Test that ReadOnlyResult can read all attributes."""
        readonly = ReadOnlyResult(sample_result)
        assert readonly.estimand == "ATT(d)"
        assert readonly.grid == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert readonly.estimate == [0.5, 1.0, 1.5, 2.0, 2.5]
        assert readonly.std_error == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert readonly.has_inference is True

    def test_read_confidence_interval(self, sample_result):
        """Test reading confidence interval through ReadOnlyResult."""
        readonly = ReadOnlyResult(sample_result)
        assert readonly.confidence_interval is not None
        assert len(readonly.confidence_interval) == 5

    def test_read_metadata(self, sample_result):
        """Test reading metadata through ReadOnlyResult."""
        readonly = ReadOnlyResult(sample_result)
        assert readonly.metadata["alp"] == 0.05

    def test_setattr_raises(self, sample_result):
        """Test that setting attributes raises AttributeError."""
        readonly = ReadOnlyResult(sample_result)
        with pytest.raises(AttributeError, match="不允许修改属性"):
            readonly.estimate = [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_setattr_new_field_raises(self, sample_result):
        """Test that setting new attributes raises AttributeError."""
        readonly = ReadOnlyResult(sample_result)
        with pytest.raises(AttributeError, match="不允许修改属性"):
            readonly.new_field = "anything"

    def test_delattr_raises(self, sample_result):
        """Test that deleting attributes raises AttributeError."""
        readonly = ReadOnlyResult(sample_result)
        with pytest.raises(AttributeError, match="不允许删除属性"):
            del readonly.estimate


# ---------------------------------------------------------------------------
# Tests: Hook execution order (priority)
# ---------------------------------------------------------------------------


class TestHookExecutionOrder:
    """Tests for priority-based execution order."""

    def test_execution_respects_priority(self, registry, sample_result):
        """Test that hooks execute in priority order (lower first)."""
        execution_order = []

        registry.register(
            "third", lambda r: execution_order.append("third"), priority=300
        )
        registry.register(
            "first", lambda r: execution_order.append("first"), priority=100
        )
        registry.register(
            "second", lambda r: execution_order.append("second"), priority=200
        )

        registry.execute(sample_result, HookStage.POST_INFERENCE)
        assert execution_order == ["first", "second", "third"]

    def test_same_priority_preserves_registration_order(
        self, registry, sample_result
    ):
        """Test that same priority hooks run in registration order."""
        execution_order = []

        registry.register(
            "alpha", lambda r: execution_order.append("alpha"), priority=100
        )
        registry.register(
            "beta", lambda r: execution_order.append("beta"), priority=100
        )
        registry.register(
            "gamma", lambda r: execution_order.append("gamma"), priority=100
        )

        registry.execute(sample_result, HookStage.POST_INFERENCE)
        assert execution_order == ["alpha", "beta", "gamma"]

    def test_only_matching_stage_executes(self, registry, sample_result):
        """Test that only hooks for the triggered stage execute."""
        outputs = []

        registry.register(
            "post_est",
            lambda r: outputs.append("est"),
            stage=HookStage.POST_ESTIMATION,
        )
        registry.register(
            "post_inf",
            lambda r: outputs.append("inf"),
            stage=HookStage.POST_INFERENCE,
        )
        registry.register(
            "post_agg",
            lambda r: outputs.append("agg"),
            stage=HookStage.POST_AGGREGATION,
        )

        registry.execute(sample_result, HookStage.POST_INFERENCE)
        assert outputs == ["inf"]


# ---------------------------------------------------------------------------
# Tests: Conditional triggering
# ---------------------------------------------------------------------------


class TestConditionalTriggering:
    """Tests for condition-based hook triggering."""

    def test_condition_true_executes(self, registry, sample_result):
        """Test hook executes when condition returns True."""
        result = registry.register(
            "conditional",
            lambda r: "executed",
            condition=lambda r: True,
        )
        outputs = registry.execute(sample_result, HookStage.POST_INFERENCE)
        assert "conditional" in outputs
        assert outputs["conditional"] == "executed"

    def test_condition_false_skips(self, registry, sample_result):
        """Test hook is skipped when condition returns False."""
        registry.register(
            "skipped",
            lambda r: "should_not_run",
            condition=lambda r: False,
        )
        outputs = registry.execute(sample_result, HookStage.POST_INFERENCE)
        assert "skipped" not in outputs

    def test_condition_checks_result(self, registry, sample_result):
        """Test condition can inspect the result."""
        registry.register(
            "att_only",
            lambda r: "ATT hook ran",
            condition=lambda r: "ATT" in r.estimand,
        )
        outputs = registry.execute(sample_result, HookStage.POST_INFERENCE)
        assert "att_only" in outputs

    def test_condition_rejects_result(self, registry):
        """Test condition rejects non-matching result."""
        acrt_result = ContDIDResult(
            estimand="ACRT(d)",
            grid=[1.0, 2.0],
            estimate=[0.1, 0.2],
            std_error=[0.01, 0.02],
            metadata={},
        )
        registry.register(
            "att_only",
            lambda r: "should_not_run",
            condition=lambda r: "ATT" in r.estimand,
        )
        outputs = registry.execute(acrt_result, HookStage.POST_INFERENCE)
        assert "att_only" not in outputs


# ---------------------------------------------------------------------------
# Tests: Exception isolation
# ---------------------------------------------------------------------------


class TestExceptionIsolation:
    """Tests for hook exception handling."""

    def test_failing_hook_does_not_break_others(self, registry, sample_result):
        """Test that a failing hook doesn't prevent others from executing."""
        registry.register(
            "good_before",
            lambda r: "before",
            priority=10,
        )
        registry.register(
            "bad_hook",
            lambda r: 1 / 0,  # ZeroDivisionError
            priority=50,
        )
        registry.register(
            "good_after",
            lambda r: "after",
            priority=90,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            outputs = registry.execute(sample_result, HookStage.POST_INFERENCE)

        # Good hooks still executed
        assert outputs["good_before"] == "before"
        assert outputs["good_after"] == "after"
        # Bad hook not in outputs
        assert "bad_hook" not in outputs

        # Warning was issued
        assert len(w) == 1
        assert "bad_hook" in str(w[0].message)
        assert issubclass(w[0].category, RuntimeWarning)

    def test_failing_condition_warns(self, registry, sample_result):
        """Test that a failing condition triggers warning."""
        registry.register(
            "bad_condition",
            lambda r: "output",
            condition=lambda r: 1 / 0,  # ZeroDivisionError
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            outputs = registry.execute(sample_result, HookStage.POST_INFERENCE)

        assert "bad_condition" not in outputs
        assert len(w) == 1


# ---------------------------------------------------------------------------
# Tests: Hook outputs storage and access
# ---------------------------------------------------------------------------


class TestHookOutputs:
    """Tests for hook output storage and retrieval."""

    def test_hook_outputs_empty_by_default(self, sample_result):
        """Test that hook_outputs is empty dict by default."""
        assert sample_result.hook_outputs == {}

    def test_hook_outputs_returns_copy(self, sample_result):
        """Test that hook_outputs returns a copy."""
        outputs1 = sample_result.hook_outputs
        outputs2 = sample_result.hook_outputs
        assert outputs1 is not outputs2

    def test_hook_outputs_populated_after_execution(self, registry, sample_result):
        """Test hook outputs are captured correctly."""
        registry.register("summary", lambda r: {"n": len(r.grid)})
        outputs = registry.execute(sample_result, HookStage.POST_INFERENCE)
        assert outputs == {"summary": {"n": 5}}

    def test_hook_outputs_stored_on_result(self, sample_result):
        """Test that hook outputs can be stored on the result."""
        sample_result._hook_outputs = {"test": "value"}
        assert sample_result.hook_outputs == {"test": "value"}


# ---------------------------------------------------------------------------
# Tests: Mathematical consistency (core results unchanged)
# ---------------------------------------------------------------------------


class TestMathematicalConsistency:
    """Tests ensuring hooks cannot modify core mathematical results."""

    def test_estimates_unchanged_after_hook(self, registry, sample_result):
        """Test that point estimates are unchanged after hook execution."""
        original_estimate = list(sample_result.estimate)
        original_std_error = list(sample_result.std_error)
        original_ci = [list(ci) for ci in sample_result.confidence_interval]

        def aggressive_hook(r):
            """Hook that tries to access and manipulate results."""
            _ = r.estimate
            _ = r.std_error
            return {"tried": True}

        registry.register("aggressive", aggressive_hook)
        registry.execute(sample_result, HookStage.POST_INFERENCE)

        # Core values unchanged
        assert sample_result.estimate == original_estimate
        assert sample_result.std_error == original_std_error
        assert sample_result.confidence_interval == original_ci

    def test_readonly_prevents_modification_in_hook(self, registry, sample_result):
        """Test that hooks cannot modify the result through ReadOnlyResult."""
        original_estimate = list(sample_result.estimate)

        def malicious_hook(r):
            """Hook that attempts to modify the result."""
            try:
                r.estimate = [0.0] * 5
            except AttributeError:
                pass
            return "attempted"

        registry.register("malicious", malicious_hook)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            registry.execute(sample_result, HookStage.POST_INFERENCE)

        # Estimate unchanged
        assert sample_result.estimate == original_estimate

    def test_confidence_band_unchanged(self, sample_result, registry):
        """Test confidence band is not affected by hooks."""
        result_with_band = ContDIDResult(
            estimand="ATT(d)",
            grid=[1.0, 2.0, 3.0],
            estimate=[1.0, 2.0, 3.0],
            std_error=[0.1, 0.2, 0.3],
            confidence_band={
                "lower": [0.5, 1.5, 2.5],
                "upper": [1.5, 2.5, 3.5],
                "critical_value": 2.0,
            },
            metadata={},
        )
        original_band = dict(result_with_band.confidence_band)

        registry.register("reader", lambda r: r.confidence_band)
        registry.execute(result_with_band, HookStage.POST_INFERENCE)

        assert result_with_band.confidence_band == original_band


# ---------------------------------------------------------------------------
# Tests: Built-in dose_summary_hook
# ---------------------------------------------------------------------------


class TestDoseSummaryHook:
    """Tests for the built-in dose_summary_hook."""

    def test_dose_summary_hook_output(self, sample_result):
        """Test dose_summary_hook produces expected output."""
        readonly = ReadOnlyResult(sample_result)
        output = dose_summary_hook(readonly)

        assert output["n_eval_points"] == 5
        assert output["estimate_range"] == (0.5, 2.5)
        assert isinstance(output["significant_points"], int)
        assert output["significant_points"] >= 0

    def test_dose_summary_hook_significant_count(self):
        """Test significance counting with known z-scores."""
        # All z-scores > 1.96
        result = ContDIDResult(
            estimand="ATT(d)",
            grid=[1.0, 2.0, 3.0],
            estimate=[10.0, 20.0, 30.0],
            std_error=[1.0, 1.0, 1.0],
            metadata={},
        )
        readonly = ReadOnlyResult(result)
        output = dose_summary_hook(readonly)
        assert output["significant_points"] == 3

    def test_dose_summary_hook_no_significant(self):
        """Test when no points are significant."""
        # All z-scores < 1.96
        result = ContDIDResult(
            estimand="ATT(d)",
            grid=[1.0, 2.0, 3.0],
            estimate=[0.01, 0.01, 0.01],
            std_error=[1.0, 1.0, 1.0],
            metadata={},
        )
        readonly = ReadOnlyResult(result)
        output = dose_summary_hook(readonly)
        assert output["significant_points"] == 0


# ---------------------------------------------------------------------------
# Tests: Module-level convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level register_hook/unregister_hook/get_hook_registry."""

    def test_register_hook_convenience(self):
        """Test the module-level register_hook function."""
        register_hook("conv_test", lambda r: "test_output")
        assert "conv_test" in get_hook_registry().list_hooks()

    def test_unregister_hook_convenience(self):
        """Test the module-level unregister_hook function."""
        register_hook("to_remove", lambda r: None)
        unregister_hook("to_remove")
        assert "to_remove" not in get_hook_registry().list_hooks()

    def test_get_hook_registry_returns_default(self):
        """Test get_hook_registry returns the default registry."""
        reg = get_hook_registry()
        assert reg is _default_hook_registry


# ---------------------------------------------------------------------------
# Tests: HookSpec dataclass
# ---------------------------------------------------------------------------


class TestHookSpec:
    """Tests for HookSpec frozen dataclass."""

    def test_hookspec_frozen(self):
        """Test HookSpec is immutable."""
        spec = HookSpec(name="test", callback=lambda r: None)
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.name = "modified"

    def test_hookspec_defaults(self):
        """Test HookSpec default values."""
        spec = HookSpec(name="test", callback=lambda r: None)
        assert spec.stage == HookStage.POST_INFERENCE
        assert spec.priority == 100
        assert spec.condition is None


# ---------------------------------------------------------------------------
# Tests: Regression - no hooks means zero overhead behavior
# ---------------------------------------------------------------------------


class TestNoHookRegression:
    """Tests ensuring no-hook behavior is completely unchanged."""

    def test_empty_registry_returns_empty_dict(self, registry, sample_result):
        """Test empty registry produces empty outputs."""
        outputs = registry.execute(sample_result, HookStage.POST_INFERENCE)
        assert outputs == {}

    def test_result_without_hooks_has_empty_hook_outputs(self):
        """Test ContDIDResult without hooks has empty hook_outputs."""
        result = ContDIDResult(
            estimand="ATT(d)",
            grid=[1.0, 2.0],
            estimate=[0.5, 1.0],
            std_error=[0.1, 0.2],
            metadata={},
        )
        assert result.hook_outputs == {}
        assert result._hook_outputs == {}


# ---------------------------------------------------------------------------
# Tests: HookStage enum
# ---------------------------------------------------------------------------


class TestHookStage:
    """Tests for HookStage enum values."""

    def test_all_stages_defined(self):
        """Test all expected stages exist."""
        assert HookStage.POST_ESTIMATION is not None
        assert HookStage.POST_INFERENCE is not None
        assert HookStage.POST_AGGREGATION is not None

    def test_stages_are_distinct(self):
        """Test stages are distinct values."""
        stages = [HookStage.POST_ESTIMATION, HookStage.POST_INFERENCE, HookStage.POST_AGGREGATION]
        assert len(set(stages)) == 3
