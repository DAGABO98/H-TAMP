from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import numpy as np


@dataclass
class WeightState:
    """
    Smoothed online reliability state.

    S_over:
        Smoothed true-overprediction score.

    S_time:
        Smoothed timing-error score.
    """
    S_over: float = 0.0
    S_time: float = 0.0


@dataclass
class WeightConfig:
    """
    Hyperparameters for the prediction weight.

    alpha_over:
        EWMA update rate for true-overprediction score.

    alpha_time:
        EWMA update rate for timing-error score.

    beta_over:
        Strength of true-overprediction penalty.

    beta_time:
        Strength of timing-error penalty.
        Usually beta_time < beta_over.

    lambda_min:
        Minimum allowed future-prediction weight.

    ignore_zero_prediction_windows:
        If True, windows with no predicted request mass do not update the
        weight. This prevents sparse empty windows and missed requests from
        increasing or decreasing a multiplier that only applies to predicted
        future request costs.

    eps:
        Small numerical constant.
    """
    alpha_over: float = 0.1
    alpha_time: float = 0.1
    beta_over: float = 2.0
    beta_time: float = 0.5
    lambda_min: float = 0.1
    ignore_zero_prediction_windows: bool = True
    eps: float = 1e-6


@dataclass(frozen=True)
class PredictionWeightKey:
    """
    Identity for one maintained prediction reliability weight.

    A patient/floor/day owns two possible keys over the horizon: one for
    monitoring requests and one for delivery requests.
    """
    patient_id: str
    floor: Optional[int]
    day: str
    request_family: str


class PredictionWeightTracker:
    """
    Maintains online prediction weights over fixed time bins.

    The tracker records one prediction snapshot per bin. Snapshot k stores
    predicted counts for bins k and k+1, then updates weight k only after the
    end of bin k+1 has been reached. The two-bin window lets the single-weight
    update function give partial credit to predictions that are off by one bin.
    """

    MONITORING_FAMILY = "monitoring"
    DELIVERY_FAMILY = "delivery"

    def __init__(
        self,
        *,
        bin_minutes: float = 5.0,
        config: Optional[WeightConfig] = None,
        default_floor: Optional[int] = None,
        default_day: str = "",
    ) -> None:
        if bin_minutes <= 0:
            raise ValueError("bin_minutes must be positive.")

        self.bin_size_seconds = float(bin_minutes) * 60.0
        self.config = config or WeightConfig()
        self.default_floor = default_floor
        self.default_day = str(default_day or "")

        self.states: Dict[PredictionWeightKey, WeightState] = {}
        self.weights: Dict[PredictionWeightKey, float] = {}
        self.diagnostics: Dict[Tuple[PredictionWeightKey, int], Dict[str, np.ndarray | float]] = {}

        self._observed_counts: Dict[Tuple[PredictionWeightKey, int], float] = {}
        self._observed_request_ids: set[str] = set()
        self._prediction_sample_counts: Dict[int, int] = {}
        self._prediction_windows: Dict[int, Dict[PredictionWeightKey, np.ndarray]] = {}
        self._recorded_prediction_bins: set[int] = set()
        self._updated_prediction_bins: set[int] = set()

    def _bin_index(self, time_seconds: float) -> int:
        return int(float(time_seconds) // self.bin_size_seconds)

    @staticmethod
    def _normalize_identifier(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return "" if text.lower() in {"", "nan", "none", "<na>"} else text

    @staticmethod
    def _normalize_floor(value: object, default_floor: Optional[int]) -> Optional[int]:
        if value is None or value == "":
            return default_floor
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default_floor

    @staticmethod
    def _request_family(request: object) -> Optional[str]:
        request_type = str(getattr(request, "request_type", "")).strip()
        if not request_type:
            return None
        if request_type == "medication":
            return PredictionWeightTracker.DELIVERY_FAMILY
        return PredictionWeightTracker.MONITORING_FAMILY

    @staticmethod
    def _iter_requests_from_requests_lists(requests_lists: object) -> Iterable[object]:
        if requests_lists is None:
            return
        dataclass_fields = getattr(requests_lists, "__dataclass_fields__", {})
        for data_field in dataclass_fields:
            yield from getattr(requests_lists, data_field)

    @classmethod
    def _iter_requests_from_sample_set(cls, prediction_sample_set: dict[float, object]) -> Iterable[object]:
        for requests_lists in prediction_sample_set.values():
            yield from cls._iter_requests_from_requests_lists(requests_lists)

    def key_for_request(self, request: object) -> Optional[PredictionWeightKey]:
        request_family = self._request_family(request)
        if request_family is None:
            return None

        patient_id = self._normalize_identifier(getattr(request, "patient_id", ""))
        if not patient_id:
            return None

        floor = self._normalize_floor(getattr(request, "floor", None), self.default_floor)
        day = self._normalize_identifier(getattr(request, "scheduled_day", "")) or self.default_day
        return PredictionWeightKey(
            patient_id=patient_id,
            floor=floor,
            day=day,
            request_family=request_family,
        )

    def record_observed_requests(self, requests_lists: object) -> None:
        """
        Add newly observed real requests to their scheduled-time bins.

        Requests are counted once by request_id so repeated policy calls with
        the same request list do not inflate the observed counts.
        """
        for request in self._iter_requests_from_requests_lists(requests_lists):
            request_id = str(getattr(request, "request_id", ""))
            if request_id and request_id in self._observed_request_ids:
                continue

            key = self.key_for_request(request)
            if key is None:
                continue

            request_bin = self._bin_index(float(getattr(request, "scheduled_time", 0.0)))
            self._observed_counts[(key, request_bin)] = (
                self._observed_counts.get((key, request_bin), 0.0) + 1.0
            )
            if request_id:
                self._observed_request_ids.add(request_id)

    def should_record_prediction_snapshot(self, current_time: float) -> bool:
        current_bin = self._bin_index(current_time)
        return current_bin not in self._recorded_prediction_bins

    def record_prediction_snapshot(
        self,
        *,
        current_time: float,
        prediction_sample_sets: list[dict[float, object]],
    ) -> None:
        """
        Record predicted counts for the current bin and the next bin.

        Each sample contributes one row in the sample-count matrix. Missing
        predictions for a key in a sample are represented as zero counts.
        """
        current_bin = self._bin_index(current_time)
        if current_bin in self._recorded_prediction_bins:
            return

        sample_count = len(prediction_sample_sets)
        if sample_count <= 0:
            self._prediction_sample_counts[current_bin] = 0
            self._prediction_windows[current_bin] = {}
            self._recorded_prediction_bins.add(current_bin)
            return

        counts_by_key: Dict[PredictionWeightKey, np.ndarray] = {}
        for sample_index, prediction_sample_set in enumerate(prediction_sample_sets):
            for request in self._iter_requests_from_sample_set(prediction_sample_set):
                key = self.key_for_request(request)
                if key is None:
                    continue
                request_bin = self._bin_index(float(getattr(request, "scheduled_time", 0.0)))
                offset = request_bin - current_bin
                if offset not in (0, 1):
                    continue
                counts = counts_by_key.setdefault(
                    key,
                    np.zeros((sample_count, 2), dtype=float),
                )
                counts[sample_index, offset] += 1.0

        self._prediction_sample_counts[current_bin] = sample_count
        self._prediction_windows[current_bin] = counts_by_key
        self._recorded_prediction_bins.add(current_bin)

    def _observed_keys_for_window(self, bin_index: int) -> set[PredictionWeightKey]:
        return {
            key
            for key, observed_bin in self._observed_counts
            if observed_bin in {bin_index, bin_index + 1}
        }

    def _observed_window_counts(self, key: PredictionWeightKey, bin_index: int) -> np.ndarray:
        return np.array(
            [
                [
                    self._observed_counts.get((key, bin_index), 0.0),
                    self._observed_counts.get((key, bin_index + 1), 0.0),
                ]
            ],
            dtype=float,
        )

    def update_due_weights(self, current_time: float) -> set[PredictionWeightKey]:
        """
        Update all bins whose following bin has fully elapsed.

        Returns the keys whose weights changed.
        """
        latest_due_bin = self._bin_index(current_time) - 2
        if latest_due_bin < 0:
            return set()

        changed_keys: set[PredictionWeightKey] = set()
        for bin_index in sorted(self._recorded_prediction_bins):
            if bin_index > latest_due_bin or bin_index in self._updated_prediction_bins:
                continue

            sample_count = self._prediction_sample_counts.get(bin_index, 0)
            if sample_count <= 0:
                self._updated_prediction_bins.add(bin_index)
                continue

            prediction_window = self._prediction_windows.get(bin_index, {})
            keys_to_update = set(prediction_window) | self._observed_keys_for_window(bin_index)
            for key in keys_to_update:
                predicted_counts = prediction_window.get(
                    key,
                    np.zeros((sample_count, 2), dtype=float),
                )
                sample_counts = predicted_counts.reshape(sample_count, 1, 2)
                observed_counts = self._observed_window_counts(key, bin_index)
                previous_weight = self.weights.get(key, 1.0)
                weight, state, diagnostics = update_prediction_weight(
                    sample_counts=sample_counts,
                    observed_counts=observed_counts,
                    state=self.states.get(key),
                    config=self.config,
                )
                self.weights[key] = weight
                self.states[key] = state
                self.diagnostics[(key, bin_index)] = diagnostics
                if abs(weight - previous_weight) > self.config.eps:
                    changed_keys.add(key)

            self._updated_prediction_bins.add(bin_index)

        return changed_keys

    def weight_for_request(self, request: object) -> float:
        key = self.key_for_request(request)
        if key is None:
            return 1.0
        return self.weights.get(key, 1.0)


def validate_inputs(
    sample_counts: np.ndarray,
    observed_counts: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Validate input shapes.

    Parameters
    ----------
    sample_counts:
        Array of shape [K, B, H].
        K = number of sampled futures.
        B = number of spatial/type bins.
        H = number of lead-time buckets.

    observed_counts:
        Array of shape [B, H].

    Returns
    -------
    K, B, H
    """
    sample_counts = np.asarray(sample_counts, dtype=float)
    observed_counts = np.asarray(observed_counts, dtype=float)

    if sample_counts.ndim != 3:
        raise ValueError(
            f"sample_counts must have shape [K, B, H], got {sample_counts.shape}"
        )

    if observed_counts.ndim != 2:
        raise ValueError(
            f"observed_counts must have shape [B, H], got {observed_counts.shape}"
        )

    K, B, H = sample_counts.shape

    if observed_counts.shape != (B, H):
        raise ValueError(
            "observed_counts shape must match sample_counts[1:]. "
            f"Expected {(B, H)}, got {observed_counts.shape}"
        )

    if K <= 0:
        raise ValueError("sample_counts must contain at least one sampled future.")

    if np.any(sample_counts < 0):
        raise ValueError("sample_counts must be nonnegative.")

    if np.any(observed_counts < 0):
        raise ValueError("observed_counts must be nonnegative.")

    return K, B, H


def empirical_surprise_scores(
    sample_counts: np.ndarray,
    observed_counts: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical overprediction and underprediction surprise scores.

    For each bin b and bucket k:

        p_over[b,k]
            Small when observed count is unusually LOW compared with samples.

        p_under[b,k]
            Small when observed count is unusually HIGH compared with samples.

    Then:

        E_over  = -log(p_over)
        E_under = -log(p_under)

    Parameters
    ----------
    sample_counts:
        Shape [K, B, H].

    observed_counts:
        Shape [B, H].

    eps:
        Numerical floor for p-values.

    Returns
    -------
    E_over:
        Shape [B, H].

    E_under:
        Shape [B, H].

    p_over:
        Shape [B, H].

    p_under:
        Shape [B, H].
    """
    K, B, H = validate_inputs(sample_counts, observed_counts)

    E_over = np.zeros((B, H), dtype=float)
    E_under = np.zeros((B, H), dtype=float)
    p_over = np.zeros((B, H), dtype=float)
    p_under = np.zeros((B, H), dtype=float)

    for b in range(B):
        for k in range(H):
            samples = sample_counts[:, b, k]
            observed = observed_counts[b, k]

            # Small p_over means observed count was unusually low.
            p_o = (1.0 + np.sum(samples <= observed)) / (K + 1.0)

            # Small p_under means observed count was unusually high.
            p_u = (1.0 + np.sum(samples >= observed)) / (K + 1.0)

            p_over[b, k] = p_o
            p_under[b, k] = p_u

            E_over[b, k] = -np.log(max(p_o, eps))
            E_under[b, k] = -np.log(max(p_u, eps))

    return E_over, E_under, p_over, p_under


def match_adjacent_L1(
    mu: np.ndarray,
    observed_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    L=1 timing matching.

    This matches predicted excess and observed excess only between adjacent
    lead-time buckets.

    Convention:
        Left-to-right greedy.
        For each adjacent pair (k, k+1), first match:

            predicted-too-early:
                predicted excess in k
                observed excess in k+1

        Then match:

            predicted-too-late:
                predicted excess in k+1
                observed excess in k

    Parameters
    ----------
    mu:
        Mean predicted counts, shape [B, H].

    observed_counts:
        Observed counts, shape [B, H].

    Returns
    -------
    true_over:
        Shape [B, H].
        Remaining predicted excess after timing matching.
        This is penalized strongly.

    true_under:
        Shape [B, H].
        Remaining observed excess after timing matching.
        This is ignored by the weight function.

    matched_timing:
        Shape [B, H, H].
        matched_timing[b, i, j] is the amount of demand predicted in bucket i
        and observed in bucket j.

        If i < j: predicted too early.
        If i > j: predicted too late.
    """
    mu = np.asarray(mu, dtype=float)
    observed_counts = np.asarray(observed_counts, dtype=float)

    if mu.shape != observed_counts.shape:
        raise ValueError(
            f"mu and observed_counts must have the same shape. "
            f"Got {mu.shape} and {observed_counts.shape}."
        )

    if mu.ndim != 2:
        raise ValueError(f"mu must have shape [B, H], got {mu.shape}")

    B, H = mu.shape

    predicted_excess = np.maximum(mu - observed_counts, 0.0)
    observed_excess = np.maximum(observed_counts - mu, 0.0)

    rem_V = predicted_excess.copy()
    rem_U = observed_excess.copy()

    matched_timing = np.zeros((B, H, H), dtype=float)

    for b in range(B):
        for k in range(H - 1):

            # ------------------------------------------------------------
            # Case 1: predicted too early.
            #
            # The model predicted excess demand in bucket k, but observed
            # excess demand occurred one bucket later, k+1.
            # ------------------------------------------------------------
            q = min(rem_V[b, k], rem_U[b, k + 1])

            if q > 0:
                matched_timing[b, k, k + 1] += q
                rem_V[b, k] -= q
                rem_U[b, k + 1] -= q

            # ------------------------------------------------------------
            # Case 2: predicted too late.
            #
            # Observed excess demand occurred in bucket k, but the model
            # predicted excess demand one bucket later, k+1.
            # ------------------------------------------------------------
            q = min(rem_V[b, k + 1], rem_U[b, k])

            if q > 0:
                matched_timing[b, k + 1, k] += q
                rem_V[b, k + 1] -= q
                rem_U[b, k] -= q

    true_over = rem_V
    true_under = rem_U

    return true_over, true_under, matched_timing


def compute_instantaneous_scores(
    sample_counts: np.ndarray,
    observed_counts: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray | float]:
    """
    Compute instantaneous true-overprediction and timing-error scores.

    Parameters
    ----------
    sample_counts:
        Shape [K, B, H].

    observed_counts:
        Shape [B, H].

    eps:
        Numerical constant.

    Returns
    -------
    diagnostics:
        Dictionary containing:
            A_over
            A_time
            mu
            E_over
            E_under
            p_over
            p_under
            true_over
            true_under
            matched_timing
            total_predicted_mass
    """
    K, B, H = validate_inputs(sample_counts, observed_counts)

    sample_counts = np.asarray(sample_counts, dtype=float)
    observed_counts = np.asarray(observed_counts, dtype=float)

    mu = sample_counts.mean(axis=0)

    E_over, E_under, p_over, p_under = empirical_surprise_scores(
        sample_counts=sample_counts,
        observed_counts=observed_counts,
        eps=eps,
    )

    true_over, true_under, matched_timing = match_adjacent_L1(
        mu=mu,
        observed_counts=observed_counts,
    )

    total_predicted_mass = float(np.sum(mu))
    M = eps + total_predicted_mass

    # ------------------------------------------------------------
    # True-overprediction score:
    #
    # A_over = sum(true_over * over_surprise) / total_predicted_mass
    # ------------------------------------------------------------
    A_over = float(np.sum(true_over * E_over) / M)

    # ------------------------------------------------------------
    # Timing-error score:
    #
    # For each matched timing mass m[b,i,j], use the average of:
    #   - overprediction surprise in predicted-excess bucket i
    #   - underprediction surprise in observed-excess bucket j
    #
    # Since L=1, distance penalty is constant and absorbed into beta_time.
    # ------------------------------------------------------------
    timing_score_mass = 0.0

    for b in range(B):
        for i in range(H):
            for j in range(H):
                q = matched_timing[b, i, j]

                if q <= 0:
                    continue

                E_time = 0.5 * (E_over[b, i] + E_under[b, j])
                timing_score_mass += q * E_time

    A_time = float(timing_score_mass / M)

    return {
        "A_over": A_over,
        "A_time": A_time,
        "mu": mu,
        "E_over": E_over,
        "E_under": E_under,
        "p_over": p_over,
        "p_under": p_under,
        "true_over": true_over,
        "true_under": true_under,
        "matched_timing": matched_timing,
        "total_predicted_mass": total_predicted_mass,
    }


def update_prediction_weight(
    sample_counts: np.ndarray,
    observed_counts: np.ndarray,
    state: Optional[WeightState] = None,
    config: Optional[WeightConfig] = None,
) -> Tuple[float, WeightState, Dict[str, np.ndarray | float]]:
    """
    Update the online prediction weight.

    Parameters
    ----------
    sample_counts:
        Shape [K, B, H].
        Counts from sampled TPP futures.

    observed_counts:
        Shape [B, H].
        Realized counts.

    state:
        Previous WeightState. If None, initialized to zeros.

    config:
        WeightConfig. If None, default config is used.

    Returns
    -------
    lambda_t:
        Future-prediction weight in [lambda_min, 1].

    new_state:
        Updated WeightState.

    diagnostics:
        Dictionary with instantaneous scores and intermediate arrays.
    """
    if state is None:
        state = WeightState()

    if config is None:
        config = WeightConfig()

    diagnostics = compute_instantaneous_scores(
        sample_counts=sample_counts,
        observed_counts=observed_counts,
        eps=config.eps,
    )

    A_over = float(diagnostics["A_over"])
    A_time = float(diagnostics["A_time"])

    if (
        config.ignore_zero_prediction_windows
        and float(diagnostics["total_predicted_mass"]) <= config.eps
    ):
        lambda_t = (
            config.lambda_min
            + (1.0 - config.lambda_min)
            * np.exp(
                -config.beta_over * state.S_over
                -config.beta_time * state.S_time
            )
        )
        lambda_t = float(np.clip(lambda_t, config.lambda_min, 1.0))
        diagnostics["S_over"] = float(state.S_over)
        diagnostics["S_time"] = float(state.S_time)
        diagnostics["lambda_t"] = lambda_t
        diagnostics["ignored_zero_prediction_window"] = True
        return lambda_t, state, diagnostics

    S_over = (
        (1.0 - config.alpha_over) * state.S_over
        + config.alpha_over * A_over
    )

    S_time = (
        (1.0 - config.alpha_time) * state.S_time
        + config.alpha_time * A_time
    )

    lambda_t = (
        config.lambda_min
        + (1.0 - config.lambda_min)
        * np.exp(
            -config.beta_over * S_over
            -config.beta_time * S_time
        )
    )

    lambda_t = float(np.clip(lambda_t, config.lambda_min, 1.0))

    new_state = WeightState(
        S_over=float(S_over),
        S_time=float(S_time),
    )

    diagnostics["S_over"] = float(S_over)
    diagnostics["S_time"] = float(S_time)
    diagnostics["lambda_t"] = lambda_t
    diagnostics["ignored_zero_prediction_window"] = False

    return lambda_t, new_state, diagnostics
