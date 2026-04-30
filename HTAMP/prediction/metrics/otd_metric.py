from __future__ import annotations

from dataclasses import dataclass
from math import inf
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

Number = Union[int, float]
Mark = Union[int, float, str, Sequence[Number]]


@dataclass(frozen=True)
class Event:
    """A single marked event in a temporal point-process rollout."""

    time: float
    event_type: Any
    mark: Mark


@dataclass(frozen=True)
class MOTDConfig:
    """Configuration for marked OTD.

    mark_mode may be one of:
        - categorical: exact mark match/mismatch
        - discrete_bin: ordered bins with normalized absolute distance
        - continuous_l1: scaled L1 distance
        - continuous_l2: scaled Euclidean distance
    """

    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.5
    c_del: float = 1.0
    c_ins: float = 1.0
    default_tau: float = 1.0
    mark_mode: str = "categorical"
    num_bins: Optional[int] = None
    mark_scale: Optional[Sequence[float]] = None
    hard_type: bool = True
    type_subst_matrix: Optional[Mapping[Any, Mapping[Any, float]]] = None


@dataclass(frozen=True)
class MOTDCostBreakdown:
    """Decomposed cost for one sequence-pair OTD computation."""

    total: float = 0.0
    time: float = 0.0
    type: float = 0.0
    mark: float = 0.0
    edit: float = 0.0
    delete: float = 0.0
    insert: float = 0.0

    def __add__(self, other: "MOTDCostBreakdown") -> "MOTDCostBreakdown":
        return MOTDCostBreakdown(
            total=self.total + other.total,
            time=self.time + other.time,
            type=self.type + other.type,
            mark=self.mark + other.mark,
            edit=self.edit + other.edit,
            delete=self.delete + other.delete,
            insert=self.insert + other.insert,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total),
            "time": float(self.time),
            "type": float(self.type),
            "mark": float(self.mark),
            "edit": float(self.edit),
            "delete": float(self.delete),
            "insert": float(self.insert),
        }


@dataclass(frozen=True)
class MOTDResult:
    """Result for a single sequence-pair MOTD computation."""

    cost: float
    alignment: List[Tuple[int, int]]
    operations: List[Tuple[str, Optional[int], Optional[int]]]
    time_cost: float = 0.0
    type_cost: float = 0.0
    mark_cost: float = 0.0
    edit_cost: float = 0.0
    delete_cost: float = 0.0
    insert_cost: float = 0.0

    @property
    def other_cost(self) -> float:
        return float(self.type_cost + self.mark_cost + self.edit_cost)

    @property
    def breakdown(self) -> MOTDCostBreakdown:
        return MOTDCostBreakdown(
            total=self.cost,
            time=self.time_cost,
            type=self.type_cost,
            mark=self.mark_cost,
            edit=self.edit_cost,
            delete=self.delete_cost,
            insert=self.insert_cost,
        )


@dataclass(frozen=True)
class ExpectedMOTDResult:
    """Monte Carlo aggregate over multiple sampled rollouts."""

    mean_cost: float
    per_sample_costs: List[float]
    mean_time_cost: float = 0.0
    mean_type_cost: float = 0.0
    mean_mark_cost: float = 0.0
    mean_edit_cost: float = 0.0
    mean_delete_cost: float = 0.0
    mean_insert_cost: float = 0.0
    per_sample_breakdowns: List[MOTDCostBreakdown] | None = None

    @property
    def mean_other_cost(self) -> float:
        return float(self.mean_type_cost + self.mean_mark_cost + self.mean_edit_cost)


TimeScales = Union[Sequence[Number], Mapping[Any, Sequence[Number]], None]


def _validate_config(config: MOTDConfig) -> None:
    if config.alpha < 0 or config.beta < 0 or config.gamma < 0:
        raise ValueError("alpha, beta, and gamma must be non-negative.")
    if config.c_del < 0 or config.c_ins < 0:
        raise ValueError("c_del and c_ins must be non-negative.")
    if config.default_tau <= 0:
        raise ValueError("default_tau must be positive.")
    if config.mark_mode == "discrete_bin":
        if config.num_bins is None or config.num_bins < 2:
            raise ValueError("num_bins must be >= 2 for discrete_bin mode.")
    elif config.mark_mode not in {"categorical", "continuous_l1", "continuous_l2"}:
        raise ValueError(
            "mark_mode must be one of: categorical, discrete_bin, "
            "continuous_l1, continuous_l2."
        )


def _as_vector(mark: Mark) -> List[float]:
    if isinstance(mark, str):
        raise TypeError("String marks are only supported with mark_mode='categorical'.")
    if isinstance(mark, (int, float)):
        return [float(mark)]
    return [float(x) for x in mark]


def mark_distance(mark_pred: Mark, mark_true: Mark, config: MOTDConfig) -> float:
    """Compute mark distance."""
    if config.mark_mode == "categorical":
        return 0.0 if mark_pred == mark_true else 1.0

    if config.mark_mode == "discrete_bin":
        assert config.num_bins is not None
        pred_bin = float(mark_pred)
        true_bin = float(mark_true)
        return abs(pred_bin - true_bin) / (config.num_bins - 1)

    pred_vec = _as_vector(mark_pred)
    true_vec = _as_vector(mark_true)
    if len(pred_vec) != len(true_vec):
        raise ValueError("Continuous marks must have the same dimensionality.")

    if config.mark_scale is None:
        scales = [1.0] * len(pred_vec)
    else:
        if len(config.mark_scale) != len(pred_vec):
            raise ValueError("mark_scale must match the continuous mark dimensionality.")
        scales = [max(float(scale), 1e-12) for scale in config.mark_scale]

    diffs = [
        (pred_value - true_value) / scale
        for pred_value, true_value, scale in zip(pred_vec, true_vec, scales)
    ]

    if config.mark_mode == "continuous_l1":
        return sum(abs(diff) for diff in diffs)

    return sum(diff * diff for diff in diffs) ** 0.5


def get_tau(
    time_scales: TimeScales,
    event_type: Any,
    default_tau: float,
) -> float:
    """Resolve the time normalization scale tau."""
    if time_scales is None:
        return default_tau

    if isinstance(time_scales, Mapping):
        gaps = list(time_scales.get(event_type, []))
    else:
        gaps = list(time_scales)

    if not gaps:
        return default_tau

    med = float(median(gaps))
    return max(med, 1e-12)


def type_substitution_cost(pred_type: Any, true_type: Any, config: MOTDConfig) -> float:
    """Compute beta * Delta_d for soft-type mode."""
    if pred_type == true_type:
        return 0.0

    if config.type_subst_matrix is not None:
        try:
            raw_cost = config.type_subst_matrix[pred_type][true_type]
        except KeyError as exc:
            raise KeyError(
                f"Missing substitution cost for ({pred_type!r}, {true_type!r})."
            ) from exc
        return config.beta * float(raw_cost)

    return config.beta


def match_cost_breakdown(
    pred_event: Event,
    true_event: Event,
    tau: float,
    config: MOTDConfig,
) -> MOTDCostBreakdown:
    """Compute decomposed substitution cost s(i, j)."""
    if config.hard_type and pred_event.event_type != true_event.event_type:
        return MOTDCostBreakdown(total=inf)

    time_cost = config.alpha * abs(pred_event.time - true_event.time) / max(tau, 1e-12)
    mark_cost = config.gamma * mark_distance(pred_event.mark, true_event.mark, config)
    type_cost = 0.0 if config.hard_type else type_substitution_cost(
        pred_event.event_type,
        true_event.event_type,
        config,
    )
    return MOTDCostBreakdown(
        total=time_cost + type_cost + mark_cost,
        time=time_cost,
        type=type_cost,
        mark=mark_cost,
    )


def match_cost(pred_event: Event, true_event: Event, tau: float, config: MOTDConfig) -> float:
    """Compute total substitution cost s(i, j) for one aligned event pair."""
    return match_cost_breakdown(
        pred_event=pred_event,
        true_event=true_event,
        tau=tau,
        config=config,
    ).total


def marked_otd(
    pred_seq: Sequence[Event],
    true_seq: Sequence[Event],
    config: Optional[MOTDConfig] = None,
    time_scales: TimeScales = None,
    return_alignment: bool = True,
) -> MOTDResult:
    """Compute one-sample Marked OTD via ordered dynamic programming."""
    cfg = config or MOTDConfig()
    _validate_config(cfg)

    n_pred = len(pred_seq)
    n_true = len(true_seq)

    dp: List[List[float]] = [[0.0] * (n_true + 1) for _ in range(n_pred + 1)]
    back: List[List[Tuple[str, int, int]]] = [
        [("", 0, 0)] * (n_true + 1) for _ in range(n_pred + 1)
    ]

    for i in range(1, n_pred + 1):
        dp[i][0] = i * cfg.c_del
        back[i][0] = ("delete", i - 1, 0)

    for j in range(1, n_true + 1):
        dp[0][j] = j * cfg.c_ins
        back[0][j] = ("insert", 0, j - 1)

    for i in range(1, n_pred + 1):
        pred_event = pred_seq[i - 1]
        for j in range(1, n_true + 1):
            true_event = true_seq[j - 1]
            tau = get_tau(time_scales, true_event.event_type, cfg.default_tau)
            sub_cost = match_cost(pred_event, true_event, tau, cfg)

            delete_val = dp[i - 1][j] + cfg.c_del
            insert_val = dp[i][j - 1] + cfg.c_ins
            match_val = dp[i - 1][j - 1] + sub_cost

            best_val = min(delete_val, insert_val, match_val)
            dp[i][j] = best_val

            if best_val == match_val:
                back[i][j] = ("match", i - 1, j - 1)
            elif best_val == delete_val:
                back[i][j] = ("delete", i - 1, j)
            else:
                back[i][j] = ("insert", i, j - 1)

    alignment: List[Tuple[int, int]] = []
    operations: List[Tuple[str, Optional[int], Optional[int]]] = []
    breakdown = MOTDCostBreakdown()
    i, j = n_pred, n_true

    while i > 0 or j > 0:
        action, next_i, next_j = back[i][j]

        if action == "match":
            pred_idx = i - 1
            true_idx = j - 1
            pred_event = pred_seq[pred_idx]
            true_event = true_seq[true_idx]
            tau = get_tau(time_scales, true_event.event_type, cfg.default_tau)
            breakdown = breakdown + match_cost_breakdown(
                pred_event=pred_event,
                true_event=true_event,
                tau=tau,
                config=cfg,
            )
            if return_alignment:
                alignment.append((pred_idx, true_idx))
                operations.append(("match", pred_idx, true_idx))
        elif action == "delete":
            breakdown = breakdown + MOTDCostBreakdown(
                total=cfg.c_del,
                edit=cfg.c_del,
                delete=cfg.c_del,
            )
            if return_alignment:
                operations.append(("delete", i - 1, None))
        elif action == "insert":
            breakdown = breakdown + MOTDCostBreakdown(
                total=cfg.c_ins,
                edit=cfg.c_ins,
                insert=cfg.c_ins,
            )
            if return_alignment:
                operations.append(("insert", None, j - 1))
        else:
            raise RuntimeError("Invalid backpointer encountered during backtracking.")

        i, j = next_i, next_j

    alignment.reverse()
    operations.reverse()
    return MOTDResult(
        cost=dp[n_pred][n_true],
        alignment=alignment if return_alignment else [],
        operations=operations if return_alignment else [],
        time_cost=breakdown.time,
        type_cost=breakdown.type,
        mark_cost=breakdown.mark,
        edit_cost=breakdown.edit,
        delete_cost=breakdown.delete,
        insert_cost=breakdown.insert,
    )


def expected_marked_otd(
    sampled_pred_sequences: Sequence[Sequence[Event]],
    true_sequence: Sequence[Event],
    config: Optional[MOTDConfig] = None,
    time_scales: TimeScales = None,
) -> ExpectedMOTDResult:
    """Monte Carlo estimate of Expected Marked OTD over sampled rollouts."""
    if not sampled_pred_sequences:
        raise ValueError("sampled_pred_sequences must contain at least one rollout.")

    results = [
        marked_otd(
            pred_seq=rollout,
            true_seq=true_sequence,
            config=config,
            time_scales=time_scales,
            return_alignment=False,
        )
        for rollout in sampled_pred_sequences
    ]
    sample_count = len(results)
    return ExpectedMOTDResult(
        mean_cost=sum(result.cost for result in results) / sample_count,
        per_sample_costs=[result.cost for result in results],
        mean_time_cost=sum(result.time_cost for result in results) / sample_count,
        mean_type_cost=sum(result.type_cost for result in results) / sample_count,
        mean_mark_cost=sum(result.mark_cost for result in results) / sample_count,
        mean_edit_cost=sum(result.edit_cost for result in results) / sample_count,
        mean_delete_cost=sum(result.delete_cost for result in results) / sample_count,
        mean_insert_cost=sum(result.insert_cost for result in results) / sample_count,
        per_sample_breakdowns=[result.breakdown for result in results],
    )


def _group_by_type(seq: Sequence[Event]) -> Dict[Any, List[Event]]:
    groups: Dict[Any, List[Event]] = {}
    for event in seq:
        groups.setdefault(event.event_type, []).append(event)
    return groups


def marked_otd_hard_by_type(
    pred_seq: Sequence[Event],
    true_seq: Sequence[Event],
    config: Optional[MOTDConfig] = None,
    time_scales: TimeScales = None,
) -> float:
    """Exact speedup for hard-type matching."""
    cfg = config or MOTDConfig(hard_type=True)
    if not cfg.hard_type:
        raise ValueError("marked_otd_hard_by_type requires hard_type=True.")

    pred_groups = _group_by_type(pred_seq)
    true_groups = _group_by_type(true_seq)
    event_types = set(pred_groups) | set(true_groups)

    total_cost = 0.0
    for event_type in event_types:
        result = marked_otd(
            pred_seq=pred_groups.get(event_type, []),
            true_seq=true_groups.get(event_type, []),
            config=cfg,
            time_scales=time_scales,
            return_alignment=False,
        )
        total_cost += result.cost

    return total_cost
