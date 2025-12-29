"""Unit tests for Context Management module.

Tests for:
- HandoffState dataclass (AC-10.1)
- MODEL_CONTEXT_BUDGETS constant (AC-10.2)
- fit_to_budget() iterative compression (AC-10.3)
- Trajectory injection (AC-10.4)
- Error contamination detection (AC-10.5)
"""

from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestration.context import (
    MODEL_CONTEXT_BUDGETS,
    CompressionFailedError,
    ErrorContaminationDetector,
    ErrorIssue,
    HandoffState,
    ValidationResult,
    _calculate_compression_ratio,
    _count_tokens,
    fit_to_budget,
    inject_trajectory,
)


# =============================================================================
# AC-10.1: HandoffState dataclass tests
# =============================================================================


class TestHandoffState:
    """Tests for HandoffState dataclass per ARCHITECTURE.md specification."""

    def test_handoff_state_creation_with_required_fields(self) -> None:
        """Test HandoffState can be created with required fields."""
        state = HandoffState(
            request_id="req-123",
            goal="Generate code for REST API",
            current_step=1,
            total_steps=3,
        )

        assert state.request_id == "req-123"
        assert state.goal == "Generate code for REST API"
        assert state.current_step == 1
        assert state.total_steps == 3

    def test_handoff_state_mutable_defaults_use_factory(self) -> None:
        """Test mutable defaults use field(default_factory=list) per AP-1.5."""
        # Create two instances without providing mutable fields
        state1 = HandoffState(
            request_id="req-1",
            goal="Goal 1",
            current_step=1,
            total_steps=2,
        )
        state2 = HandoffState(
            request_id="req-2",
            goal="Goal 2",
            current_step=1,
            total_steps=2,
        )

        # Mutate state1's list
        state1.constraints.append("no recursion")

        # state2's list should NOT be affected (AP-1.5 compliance)
        assert state1.constraints == ["no recursion"]
        assert state2.constraints == []  # Should be empty, not shared

    def test_handoff_state_all_list_fields_independent(self) -> None:
        """Test all mutable list fields are independent between instances."""
        state1 = HandoffState(
            request_id="req-1",
            goal="Goal",
            current_step=1,
            total_steps=1,
        )
        state2 = HandoffState(
            request_id="req-2",
            goal="Goal",
            current_step=1,
            total_steps=1,
        )

        # Mutate all list fields on state1
        state1.decisions_made.append("decision-1")
        state1.evidence_refs.append("ref-1")
        state1.active_errors.append("error-1")
        state1.resolved_errors.append("resolved-1")

        # All state2 lists should remain empty
        assert state2.decisions_made == []
        assert state2.evidence_refs == []
        assert state2.active_errors == []
        assert state2.resolved_errors == []

    def test_handoff_state_compressed_context_default_none(self) -> None:
        """Test compressed_context defaults to None (AP-1.1 compliance)."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=1,
            total_steps=1,
        )

        assert state.compressed_context is None

    def test_handoff_state_with_all_fields_provided(self) -> None:
        """Test HandoffState with all fields explicitly provided."""
        state = HandoffState(
            request_id="req-456",
            goal="Complex orchestration",
            current_step=2,
            total_steps=5,
            constraints=["memory < 8GB", "latency < 100ms"],
            decisions_made=["use llama for routing"],
            evidence_refs=["doc-1", "doc-2"],
            active_errors=["timeout warning"],
            resolved_errors=["model not found"],
            compressed_context="Previous step output...",
        )

        assert state.constraints == ["memory < 8GB", "latency < 100ms"]
        assert state.decisions_made == ["use llama for routing"]
        assert state.evidence_refs == ["doc-1", "doc-2"]
        assert state.active_errors == ["timeout warning"]
        assert state.resolved_errors == ["model not found"]
        assert state.compressed_context == "Previous step output..."

    def test_handoff_state_is_dataclass(self) -> None:
        """Test HandoffState is a proper dataclass."""
        # Should have dataclass fields
        field_names = [f.name for f in fields(HandoffState)]
        expected_fields = [
            "request_id",
            "goal",
            "current_step",
            "total_steps",
            "constraints",
            "decisions_made",
            "evidence_refs",
            "active_errors",
            "resolved_errors",
            "compressed_context",
        ]
        assert field_names == expected_fields

    def test_handoff_state_fields_have_correct_types(self) -> None:
        """Test HandoffState field type annotations."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=1,
            total_steps=3,
        )

        # Verify types at runtime
        assert isinstance(state.request_id, str)
        assert isinstance(state.goal, str)
        assert isinstance(state.current_step, int)
        assert isinstance(state.total_steps, int)
        assert isinstance(state.constraints, list)
        assert isinstance(state.decisions_made, list)
        assert isinstance(state.evidence_refs, list)
        assert isinstance(state.active_errors, list)
        assert isinstance(state.resolved_errors, list)


# =============================================================================
# AC-10.2: Token Budget Allocation tests
# =============================================================================


class TestModelContextBudgets:
    """Tests for MODEL_CONTEXT_BUDGETS constant per ARCHITECTURE.md."""

    def test_model_context_budgets_is_dict(self) -> None:
        """Test MODEL_CONTEXT_BUDGETS is a dictionary."""
        assert isinstance(MODEL_CONTEXT_BUDGETS, dict)

    def test_llama_3_2_3b_budget(self) -> None:
        """Test llama-3.2-3b has correct budget allocation."""
        budget = MODEL_CONTEXT_BUDGETS["llama-3.2-3b"]

        assert budget["total"] == 8192
        assert budget["system"] == 500
        assert budget["trajectory"] == 300
        assert budget["handoff"] == 1500
        assert budget["user_query"] == 2000
        assert budget["generation"] == 3892

    def test_granite_8b_code_128k_budget(self) -> None:
        """Test granite-8b-code-128k has correct budget allocation."""
        budget = MODEL_CONTEXT_BUDGETS["granite-8b-code-128k"]

        assert budget["total"] == 131072
        assert budget["system"] == 1000
        assert budget["trajectory"] == 500
        assert budget["handoff"] == 4000
        assert budget["user_query"] == 32000
        assert budget["generation"] == 93572

    def test_all_budgets_sum_correctly(self) -> None:
        """Test all budget allocations sum to total for each model."""
        for model_name, budget in MODEL_CONTEXT_BUDGETS.items():
            allocated = (
                budget["system"]
                + budget["trajectory"]
                + budget["handoff"]
                + budget["user_query"]
                + budget["generation"]
            )
            assert allocated == budget["total"], (
                f"{model_name}: allocated {allocated} != total {budget['total']}"
            )

    def test_all_budgets_have_required_keys(self) -> None:
        """Test all model budgets have required keys."""
        required_keys = {
            "total",
            "system",
            "trajectory",
            "handoff",
            "user_query",
            "generation",
        }

        for model_name, budget in MODEL_CONTEXT_BUDGETS.items():
            missing = required_keys - set(budget.keys())
            assert not missing, f"{model_name} missing keys: {missing}"

    def test_contains_expected_models(self) -> None:
        """Test MODEL_CONTEXT_BUDGETS contains expected model names."""
        expected_models = [
            "llama-3.2-3b",
            "granite-8b-code-128k",
        ]

        for model in expected_models:
            assert model in MODEL_CONTEXT_BUDGETS, f"Missing model: {model}"


# =============================================================================
# AC-10.3: fit_to_budget() tests
# =============================================================================


class TestFitToBudget:
    """Tests for fit_to_budget() iterative compression per AP-2.1."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock Llama model for testing."""
        model = MagicMock()
        # tokenize returns list of token IDs
        model.tokenize = MagicMock(return_value=list(range(100)))
        return model

    @pytest.mark.asyncio
    async def test_fit_to_budget_returns_content_if_under_budget(
        self, mock_model: MagicMock
    ) -> None:
        """Test content returned unchanged if already under budget."""
        content = "Short content"
        mock_model.tokenize.return_value = list(range(50))  # 50 tokens

        result = await fit_to_budget(
            content=content,
            max_tokens=100,
            model=mock_model,
        )

        assert result == content

    @pytest.mark.asyncio
    async def test_fit_to_budget_compresses_if_over_budget(
        self, mock_model: MagicMock
    ) -> None:
        """Test content is compressed if over budget."""
        content = "Very long content that needs compression"
        # First call: over budget (150 tokens)
        # Second call: under budget (80 tokens) after compression
        mock_model.tokenize.side_effect = [
            list(range(150)),  # Initial check - over budget
            list(range(80)),  # After compression - under budget
        ]

        with patch(
            "src.orchestration.context._apply_compression",
            new_callable=AsyncMock,
            return_value="Compressed content",
        ) as mock_compress:
            result = await fit_to_budget(
                content=content,
                max_tokens=100,
                model=mock_model,
            )

            assert result == "Compressed content"
            mock_compress.assert_called_once()

    @pytest.mark.asyncio
    async def test_fit_to_budget_iterates_until_under_budget(
        self, mock_model: MagicMock
    ) -> None:
        """Test iterative compression continues until under budget."""
        content = "Content requiring multiple compression rounds"
        # Simulates needing 2 iterations
        mock_model.tokenize.side_effect = [
            list(range(200)),  # Initial: 200 tokens
            list(range(150)),  # After 1st compression: 150 tokens
            list(range(90)),  # After 2nd compression: 90 tokens
        ]

        compress_results = ["First compression", "Second compression"]
        compress_mock = AsyncMock(side_effect=compress_results)

        with patch(
            "src.orchestration.context._apply_compression", compress_mock
        ):
            result = await fit_to_budget(
                content=content,
                max_tokens=100,
                model=mock_model,
            )

            assert result == "Second compression"
            assert compress_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_fit_to_budget_raises_after_max_iterations(
        self, mock_model: MagicMock
    ) -> None:
        """Test CompressionFailedError raised after max iterations."""
        content = "Content that won't fit"
        # Always over budget
        mock_model.tokenize.return_value = list(range(200))

        compress_mock = AsyncMock(return_value="Still too long")

        with (
            patch("src.orchestration.context._apply_compression", compress_mock),
            pytest.raises(CompressionFailedError) as exc_info,
        ):
            await fit_to_budget(
                content=content,
                max_tokens=100,
                model=mock_model,
                max_iterations=3,
            )

        assert "100 tokens" in str(exc_info.value)
        assert "3 iterations" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fit_to_budget_uses_iterative_not_recursive(
        self, mock_model: MagicMock
    ) -> None:
        """Test fit_to_budget uses iteration, not recursion (AP-2.1)."""
        # This test ensures the implementation uses a for loop, not recursion
        content = "Content"
        mock_model.tokenize.side_effect = [
            list(range(200)),  # Over budget
            list(range(200)),  # Still over
            list(range(200)),  # Still over
            list(range(200)),  # Still over
        ]

        compress_mock = AsyncMock(return_value="Still big")

        with (
            patch("src.orchestration.context._apply_compression", compress_mock),
            pytest.raises(CompressionFailedError),
        ):
            await fit_to_budget(
                content=content,
                max_tokens=100,
                model=mock_model,
                max_iterations=3,
            )

        # Should have exactly max_iterations calls (iterative)
        # If recursive without proper bounds, this could be more
        assert compress_mock.call_count == 3


class TestTokenHelperFunctions:
    """Tests for token counting helper functions."""

    def test_count_tokens_calls_model_tokenize(self) -> None:
        """Test _count_tokens uses model.tokenize."""
        model = MagicMock()
        model.tokenize.return_value = [1, 2, 3, 4, 5]

        result = _count_tokens("test content", model)

        assert result == 5
        model.tokenize.assert_called_once_with(b"test content")

    def test_calculate_compression_ratio_basic(self) -> None:
        """Test _calculate_compression_ratio calculates correct ratio."""
        # Target 100, current 200 -> ratio 0.5 * 0.9 = 0.45
        result = _calculate_compression_ratio(target=100, current=200)

        assert result == pytest.approx(0.45)

    def test_calculate_compression_ratio_with_safety_margin(self) -> None:
        """Test compression ratio includes 10% safety margin."""
        # Without margin: 100/200 = 0.5
        # With 10% margin: 0.5 * 0.9 = 0.45
        result = _calculate_compression_ratio(target=100, current=200)

        # Should be less than raw ratio due to safety margin
        raw_ratio = 100 / 200
        assert result < raw_ratio


# =============================================================================
# AC-10.4: Trajectory Injection tests
# =============================================================================


class TestTrajectoryInjection:
    """Tests for inject_trajectory() per ARCHITECTURE.md format."""

    def test_inject_trajectory_includes_goal(self) -> None:
        """Test trajectory includes original user goal."""
        state = HandoffState(
            request_id="req-123",
            goal="Build a REST API for user management",
            current_step=2,
            total_steps=4,
        )

        trajectory = inject_trajectory(
            state=state,
            step_name="Code Generation",
            previous_decision="Chose FastAPI framework",
            next_task="Generate endpoint handlers",
            forbidden=["Do not modify database schema"],
        )

        assert "Build a REST API for user management" in trajectory

    def test_inject_trajectory_includes_step_position(self) -> None:
        """Test trajectory includes current step position."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=2,
            total_steps=4,
        )

        trajectory = inject_trajectory(
            state=state,
            step_name="Code Generation",
            previous_decision="Previous",
            next_task="Next",
            forbidden=[],
        )

        assert "2" in trajectory
        assert "4" in trajectory

    def test_inject_trajectory_includes_step_name(self) -> None:
        """Test trajectory includes step name."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=1,
            total_steps=3,
        )

        trajectory = inject_trajectory(
            state=state,
            step_name="Architecture Design",
            previous_decision="Previous",
            next_task="Next",
            forbidden=[],
        )

        assert "Architecture Design" in trajectory

    def test_inject_trajectory_includes_previous_decision(self) -> None:
        """Test trajectory includes what was decided previously."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=2,
            total_steps=3,
        )

        trajectory = inject_trajectory(
            state=state,
            step_name="Step",
            previous_decision="Selected PostgreSQL for persistence",
            next_task="Next",
            forbidden=[],
        )

        assert "Selected PostgreSQL for persistence" in trajectory

    def test_inject_trajectory_includes_next_task(self) -> None:
        """Test trajectory includes what must be decided next."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=1,
            total_steps=2,
        )

        trajectory = inject_trajectory(
            state=state,
            step_name="Step",
            previous_decision="Previous",
            next_task="Implement authentication middleware",
            forbidden=[],
        )

        assert "Implement authentication middleware" in trajectory

    def test_inject_trajectory_includes_forbidden_actions(self) -> None:
        """Test trajectory includes forbidden actions."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=1,
            total_steps=2,
        )

        trajectory = inject_trajectory(
            state=state,
            step_name="Step",
            previous_decision="Previous",
            next_task="Next",
            forbidden=["Do not use recursion", "Avoid global state"],
        )

        assert "Do not use recursion" in trajectory
        assert "Avoid global state" in trajectory

    def test_inject_trajectory_format_matches_spec(self) -> None:
        """Test trajectory format matches ARCHITECTURE.md specification."""
        state = HandoffState(
            request_id="req-123",
            goal="Original intent",
            current_step=2,
            total_steps=4,
        )

        trajectory = inject_trajectory(
            state=state,
            step_name="Code Gen",
            previous_decision="Decision made",
            next_task="Next decision",
            forbidden=["Forbidden action"],
        )

        # Should contain section headers per spec
        assert "## Current Position" in trajectory or "Current Position" in trajectory
        assert "Goal:" in trajectory or "goal" in trajectory.lower()
        assert "Step:" in trajectory or "step" in trajectory.lower()


# =============================================================================
# AC-10.5: Error Contamination Detection tests
# =============================================================================


class TestErrorIssue:
    """Tests for ErrorIssue data structure."""

    def test_error_issue_creation(self) -> None:
        """Test ErrorIssue can be created with required fields."""
        issue = ErrorIssue(
            type="error_marker_detected",
            severity="warning",
        )

        assert issue.type == "error_marker_detected"
        assert issue.severity == "warning"

    def test_error_issue_with_optional_fields(self) -> None:
        """Test ErrorIssue with optional fields."""
        issue = ErrorIssue(
            type="contradiction_detected",
            severity="high",
            marker="I apologize",
            decision="Previous decision",
            count=3,
        )

        assert issue.marker == "I apologize"
        assert issue.decision == "Previous decision"
        assert issue.count == 3


class TestValidationResult:
    """Tests for ValidationResult data structure."""

    def test_validation_result_creation(self) -> None:
        """Test ValidationResult can be created."""
        result = ValidationResult(
            valid=True,
            issues=[],
            recommendation="proceed",
        )

        assert result.valid is True
        assert result.issues == []
        assert result.recommendation == "proceed"

    def test_validation_result_with_issues(self) -> None:
        """Test ValidationResult with issues list."""
        issues = [
            ErrorIssue(type="error_marker_detected", severity="warning"),
            ErrorIssue(type="contradiction_detected", severity="high"),
        ]

        result = ValidationResult(
            valid=False,
            issues=issues,
            recommendation="quarantine",
        )

        assert result.valid is False
        assert len(result.issues) == 2
        assert result.recommendation == "quarantine"


class TestErrorContaminationDetector:
    """Tests for ErrorContaminationDetector per ARCHITECTURE.md."""

    @pytest.fixture
    def detector(self) -> ErrorContaminationDetector:
        """Create ErrorContaminationDetector instance."""
        return ErrorContaminationDetector()

    @pytest.fixture
    def clean_state(self) -> HandoffState:
        """Create a clean HandoffState for testing."""
        return HandoffState(
            request_id="req-123",
            goal="Test goal",
            current_step=1,
            total_steps=3,
            decisions_made=["Use Python"],
        )

    @pytest.mark.asyncio
    async def test_validate_clean_output(
        self, detector: ErrorContaminationDetector, clean_state: HandoffState
    ) -> None:
        """Test validation passes for clean output."""
        output = "Here is the generated code for your REST API."

        result = detector.validate_handoff(clean_state, output)

        assert result.valid is True
        assert result.recommendation == "proceed"

    @pytest.mark.asyncio
    async def test_detect_apology_marker(
        self, detector: ErrorContaminationDetector, clean_state: HandoffState
    ) -> None:
        """Test detection of 'I apologize' error marker."""
        output = "I apologize, but I cannot generate that code."

        result = detector.validate_handoff(clean_state, output)

        assert any(i.marker == "I apologize" for i in result.issues if i.marker)

    @pytest.mark.asyncio
    async def test_detect_error_admission_marker(
        self, detector: ErrorContaminationDetector, clean_state: HandoffState
    ) -> None:
        """Test detection of 'I made an error' marker."""
        output = "I made an error in my previous response."

        result = detector.validate_handoff(clean_state, output)

        assert any(
            "error" in (i.marker or "").lower()
            for i in result.issues
        )

    @pytest.mark.asyncio
    async def test_detect_incorrect_marker(
        self, detector: ErrorContaminationDetector, clean_state: HandoffState
    ) -> None:
        """Test detection of 'That's incorrect' marker."""
        output = "That's incorrect, let me fix that."

        result = detector.validate_handoff(clean_state, output)

        assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_detect_hallucination_marker(
        self, detector: ErrorContaminationDetector, clean_state: HandoffState
    ) -> None:
        """Test detection of 'hallucination' marker."""
        output = "This might be a hallucination from the previous model."

        result = detector.validate_handoff(clean_state, output)

        assert any(
            "hallucination" in (i.marker or "").lower()
            for i in result.issues
        )

    @pytest.mark.asyncio
    async def test_detect_lack_of_information_marker(
        self, detector: ErrorContaminationDetector, clean_state: HandoffState
    ) -> None:
        """Test detection of information lacking marker."""
        output = "I don't have information about that API."

        result = detector.validate_handoff(clean_state, output)

        assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_error_marker_severity_is_warning(
        self, detector: ErrorContaminationDetector, clean_state: HandoffState
    ) -> None:
        """Test error marker issues have warning severity."""
        output = "I apologize for the confusion."

        result = detector.validate_handoff(clean_state, output)

        marker_issues = [i for i in result.issues if i.type == "error_marker_detected"]
        assert all(i.severity == "warning" for i in marker_issues)

    @pytest.mark.asyncio
    async def test_case_insensitive_marker_detection(
        self, detector: ErrorContaminationDetector, clean_state: HandoffState
    ) -> None:
        """Test marker detection is case-insensitive."""
        output = "I APOLOGIZE FOR THE ERROR."

        result = detector.validate_handoff(clean_state, output)

        assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_detect_error_accumulation(
        self, detector: ErrorContaminationDetector
    ) -> None:
        """Test detection of error accumulation."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=2,
            total_steps=3,
            active_errors=["error1", "error2", "error3"],  # 3 errors at step 2
        )

        result = detector.validate_handoff(state, "Clean output")

        accumulation_issues = [
            i for i in result.issues if i.type == "error_accumulation"
        ]
        assert len(accumulation_issues) > 0
        assert any(i.severity == "high" for i in accumulation_issues)

    @pytest.mark.asyncio
    async def test_high_severity_makes_result_invalid(
        self, detector: ErrorContaminationDetector
    ) -> None:
        """Test high severity issues make validation result invalid."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=1,
            total_steps=3,
            active_errors=["e1", "e2"],  # 2 errors at step 1 -> accumulation
        )

        result = detector.validate_handoff(state, "Output")

        assert result.valid is False
        assert result.recommendation == "quarantine"

    @pytest.mark.asyncio
    async def test_warning_only_keeps_result_valid(
        self, detector: ErrorContaminationDetector, clean_state: HandoffState
    ) -> None:
        """Test warning-only issues keep result valid."""
        output = "I apologize for the delay in processing."

        result = detector.validate_handoff(clean_state, output)

        # Has warning issues but should still be valid (no high severity)
        warning_issues = [i for i in result.issues if i.severity == "warning"]
        high_issues = [i for i in result.issues if i.severity == "high"]

        if warning_issues and not high_issues:
            assert result.valid is True

    @pytest.mark.asyncio
    async def test_error_markers_constant_has_expected_values(
        self, detector: ErrorContaminationDetector
    ) -> None:
        """Test ERROR_MARKERS contains expected values."""
        expected_markers = [
            "I apologize",
            "I made an error",
            "That's incorrect",
            "hallucination",
            "I don't have information about",
        ]

        for marker in expected_markers:
            assert marker in detector.ERROR_MARKERS


# =============================================================================
# Additional edge case tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for context management module."""

    def test_handoff_state_empty_lists_are_independent(self) -> None:
        """Test empty default lists are independent instances."""
        states = [
            HandoffState(
                request_id=f"req-{i}",
                goal="Goal",
                current_step=1,
                total_steps=1,
            )
            for i in range(3)
        ]

        # Verify all lists are different objects
        assert states[0].constraints is not states[1].constraints
        assert states[1].constraints is not states[2].constraints

    def test_handoff_state_can_be_used_in_dict(self) -> None:
        """Test HandoffState request_id can be used for dict lookup."""
        state = HandoffState(
            request_id="unique-id",
            goal="Goal",
            current_step=1,
            total_steps=1,
        )

        cache: dict[str, HandoffState] = {}
        cache[state.request_id] = state

        assert cache["unique-id"] is state

    @pytest.mark.asyncio
    async def test_fit_to_budget_with_exact_budget(self) -> None:
        """Test fit_to_budget with content exactly at budget."""
        model = MagicMock()
        model.tokenize.return_value = list(range(100))  # Exactly 100 tokens

        result = await fit_to_budget(
            content="Exact content",
            max_tokens=100,
            model=model,
        )

        assert result == "Exact content"

    def test_inject_trajectory_empty_forbidden_list(self) -> None:
        """Test inject_trajectory with empty forbidden list."""
        state = HandoffState(
            request_id="req-123",
            goal="Goal",
            current_step=1,
            total_steps=1,
        )

        # Should not raise with empty forbidden list
        trajectory = inject_trajectory(
            state=state,
            step_name="Step",
            previous_decision="Previous",
            next_task="Next",
            forbidden=[],
        )

        assert isinstance(trajectory, str)


# =============================================================================
# AC-10.6: Provider-Agnostic Compression tests
# =============================================================================


class TestProviderAgnosticCompression:
    """Tests for provider-agnostic _apply_compression per design decision.

    Design: The compression function should accept any InferenceProvider,
    allowing ANY loaded LLM to perform compression (not hardcoded to specific model).

    Reference: User requirement - "shouldn't the solution be flexible since it
    could technically be used by any LLM as long as it is loaded"
    """

    @pytest.mark.asyncio
    async def test_apply_compression_accepts_provider_parameter(self) -> None:
        """Test _apply_compression accepts InferenceProvider parameter.

        AC-10.6.1: Function signature includes provider parameter.
        """
        from src.orchestration.context import _apply_compression
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="llama-3.2-3b")
        content = "This is content that needs compression to fit budget."

        # Should accept provider parameter without error
        result = await _apply_compression(
            content=content,
            target_ratio=0.5,
            preserve=["decisions"],
            drop=["verbose"],
            provider=provider,
        )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_apply_compression_uses_provider_for_generation(self) -> None:
        """Test _apply_compression uses provider.generate() for LLM-based compression.

        AC-10.6.2: Compression delegates to provider instead of truncation.
        """
        from unittest.mock import AsyncMock, patch

        from src.orchestration.context import _apply_compression
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="compression-model")

        # Mock the generate method to return compressed content
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(message=AsyncMock(content="Compressed: key decisions only"))
        ]
        provider.generate = AsyncMock(return_value=mock_response)

        result = await _apply_compression(
            content="Verbose content with lots of reasoning chains and examples.",
            target_ratio=0.3,
            preserve=["decisions"],
            drop=["reasoning_chains", "examples"],
            provider=provider,
        )

        # Should call provider.generate() for LLM-based compression
        provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_compression_includes_preserve_categories(self) -> None:
        """Test _apply_compression prompt includes preserve categories.

        AC-10.6.3: Compression prompt instructs LLM what to preserve.
        """
        from unittest.mock import ANY, AsyncMock

        from src.orchestration.context import _apply_compression
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="llama-3.2-3b")
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Compressed"))]
        provider.generate = AsyncMock(return_value=mock_response)

        await _apply_compression(
            content="Content to compress",
            target_ratio=0.5,
            preserve=["decisions", "constraints", "errors"],
            drop=["verbose"],
            provider=provider,
        )

        # Verify the request includes preserve categories
        call_args = provider.generate.call_args[0][0]
        prompt_text = str(call_args.messages)

        assert "decisions" in prompt_text.lower() or "preserve" in prompt_text.lower()

    @pytest.mark.asyncio
    async def test_apply_compression_includes_drop_categories(self) -> None:
        """Test _apply_compression prompt includes drop categories.

        AC-10.6.4: Compression prompt instructs LLM what to drop.
        """
        from unittest.mock import AsyncMock

        from src.orchestration.context import _apply_compression
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="llama-3.2-3b")
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Compressed"))]
        provider.generate = AsyncMock(return_value=mock_response)

        await _apply_compression(
            content="Content with reasoning chains and examples",
            target_ratio=0.5,
            preserve=["decisions"],
            drop=["reasoning_chains", "examples", "verbose"],
            provider=provider,
        )

        # Verify the request includes drop categories
        call_args = provider.generate.call_args[0][0]
        prompt_text = str(call_args.messages)

        assert "drop" in prompt_text.lower() or "remove" in prompt_text.lower() or "reasoning" in prompt_text.lower()

    @pytest.mark.asyncio
    async def test_apply_compression_respects_target_ratio(self) -> None:
        """Test _apply_compression instructs LLM about target ratio.

        AC-10.6.5: Compression prompt specifies target length.
        """
        from unittest.mock import AsyncMock

        from src.orchestration.context import _apply_compression
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="llama-3.2-3b")
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Short"))]
        provider.generate = AsyncMock(return_value=mock_response)

        content = "A" * 1000  # Long content

        await _apply_compression(
            content=content,
            target_ratio=0.3,  # 30% of original
            preserve=[],
            drop=[],
            provider=provider,
        )

        # Verify target length is communicated to LLM
        call_args = provider.generate.call_args[0][0]
        prompt_text = str(call_args.messages)

        # Should mention target length (300 chars) or ratio (30%)
        assert "300" in prompt_text or "30%" in prompt_text or "ratio" in prompt_text.lower()

    @pytest.mark.asyncio
    async def test_apply_compression_works_with_any_loaded_provider(self) -> None:
        """Test _apply_compression works with different provider types.

        AC-10.6.6: Any InferenceProvider implementation can compress.
        """
        from unittest.mock import AsyncMock

        from src.orchestration.context import _apply_compression
        from tests.unit.providers.mock_provider import MockProvider

        # Test with different model configurations
        providers = [
            MockProvider(model_id="llama-3.2-3b", roles=["fast"]),
            MockProvider(model_id="phi-4", roles=["primary"]),
            MockProvider(model_id="qwen2.5-7b", roles=["coder"]),
        ]

        for provider in providers:
            mock_response = AsyncMock()
            mock_response.choices = [
                AsyncMock(message=AsyncMock(content=f"Compressed by {provider._model_id}"))
            ]
            provider.generate = AsyncMock(return_value=mock_response)

            result = await _apply_compression(
                content="Content to compress",
                target_ratio=0.5,
                preserve=["decisions"],
                drop=["verbose"],
                provider=provider,
            )

            assert provider._model_id in result

    @pytest.mark.asyncio
    async def test_apply_compression_preserves_none_defaults_per_ap_1_5(self) -> None:
        """Test _apply_compression handles None defaults correctly (AP-1.5).

        AC-10.6.7: Mutable default avoidance pattern.
        """
        from src.orchestration.context import _apply_compression
        from tests.unit.providers.mock_provider import MockProvider
        from unittest.mock import AsyncMock

        provider = MockProvider(model_id="test")
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Compressed"))]
        provider.generate = AsyncMock(return_value=mock_response)

        # Should work with default empty lists
        result = await _apply_compression(
            content="Content",
            target_ratio=0.5,
            preserve=None,  # type: ignore - testing default handling
            drop=None,  # type: ignore - testing default handling
            provider=provider,
        )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_fit_to_budget_uses_provider_for_compression(self) -> None:
        """Test fit_to_budget passes provider to _apply_compression.

        AC-10.6.8: Integration point - fit_to_budget forwards provider.
        """
        from unittest.mock import AsyncMock, patch, MagicMock

        from src.orchestration.context import fit_to_budget
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="compression-model")
        mock_model = MagicMock()
        mock_model.tokenize.side_effect = [
            list(range(200)),  # Initial: over budget
            list(range(50)),  # After compression: under budget
        ]

        with patch(
            "src.orchestration.context._apply_compression",
            new_callable=AsyncMock,
            return_value="Compressed",
        ) as mock_compress:
            result = await fit_to_budget(
                content="Long content",
                max_tokens=100,
                model=mock_model,
                provider=provider,  # New parameter
            )

            # Verify provider was passed to _apply_compression
            mock_compress.assert_called_once()
            call_kwargs = mock_compress.call_args.kwargs
            assert "provider" in call_kwargs
            assert call_kwargs["provider"] is provider