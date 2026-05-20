import datetime as dt
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dataclasses import dataclass
from typing import Optional, Iterable

IsoWeek = tuple[int, int]


# ----------------------------
# CONFIG
# ----------------------------

START = dt.date(2024, 6, 24)
END   = dt.date(2025, 6, 29)
FLOORS = [2, 3, 7, 9]
MAX_WORKERS = 100
ROLLOUT_MODES = {5, 6, 7, 8}
CACHE_POLICY_MODES = {3, *ROLLOUT_MODES}

BASE = [
    "python", "-m", "HTAMP.assignment.evaluate_assignment",
    "--use_saved_data",
    "--use_saved_request_data",
    "--save_results_csv",
    "--hour_start", "8",
    "--hour_end", "9",
]

LOG_ROOT = "results/policies/logs"


def optional_float_env(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def optional_int_env(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)

PREDICTION_CACHE_PATH = os.getenv(
    "HTAMP_PREDICTION_CACHE_PATH",
    "data/prediction/offline_request_prediction_cache_by_floor_day",
)
PREDICTION_CACHE_RUN_NAMES = os.getenv(
    "HTAMP_PREDICTION_CACHE_RUN_NAMES",
    "final_vital_sign_flex_tpp_st_enhanced_marks,final_med_flex_tpp_st_no_conditioning",
)
PREDICTION_MATCH_TOLERANCE_MINUTES = float(
    os.getenv("HTAMP_PREDICTION_MATCH_TOLERANCE_MINUTES", "10.0")
)
PREDICTION_LOOKAHEAD_MINUTES = float(
    os.getenv("HTAMP_PREDICTION_LOOKAHEAD_MINUTES", "60.0")
)
PREDICTION_WEIGHT_BIN_MINUTES = float(
    os.getenv("HTAMP_PREDICTION_WEIGHT_BIN_MINUTES", "5.0")
)
PREDICTION_WEIGHT_ALPHA_OVER = optional_float_env("HTAMP_PREDICTION_WEIGHT_ALPHA_OVER")
PREDICTION_WEIGHT_ALPHA_TIME = optional_float_env("HTAMP_PREDICTION_WEIGHT_ALPHA_TIME")
PREDICTION_WEIGHT_BETA_OVER = optional_float_env("HTAMP_PREDICTION_WEIGHT_BETA_OVER")
PREDICTION_WEIGHT_BETA_TIME = optional_float_env("HTAMP_PREDICTION_WEIGHT_BETA_TIME")
PREDICTION_WEIGHT_LAMBDA_MIN = optional_float_env("HTAMP_PREDICTION_WEIGHT_LAMBDA_MIN")
PREDICTION_WEIGHT_COLD_START_WEIGHT = optional_float_env("HTAMP_PREDICTION_WEIGHT_COLD_START_WEIGHT")
PREDICTION_WEIGHT_FALLBACK_MIN_POSITIVE_PREDICTION_WINDOWS = optional_int_env(
    "HTAMP_PREDICTION_WEIGHT_FALLBACK_MIN_POSITIVE_PREDICTION_WINDOWS"
)
PREDICTION_WEIGHT_PATIENT_CREDIBILITY_PRIOR = optional_float_env(
    "HTAMP_PREDICTION_WEIGHT_PATIENT_CREDIBILITY_PRIOR"
)
PREDICTION_CACHE_PATIENT_FILTER_MODE = os.getenv(
    "HTAMP_PREDICTION_CACHE_PATIENT_FILTER_MODE",
    "active_floor",
)
INCLUDE_FUTURE_SCHEDULED_REQUESTS = os.getenv(
    "HTAMP_INCLUDE_FUTURE_SCHEDULED_REQUESTS",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
PATIENT_ROOM_STAYS_FILE = os.getenv(
    "HTAMP_PATIENT_ROOM_STAYS_FILE",
    "data/processed/patient_room_stays.csv",
)
ADMISSIONS_DISCHARGES_FILE = os.getenv(
    "HTAMP_ADMISSIONS_DISCHARGES_FILE",
    "data/processed/admissions_discharges.csv",
)


# Only include dates whose (ISO year, ISO week) is in this explicit list
ALLOWED_ISO_WEEKS: set[IsoWeek] = {
    # Weeks with highest demand
    (2024, 40),
    (2025, 5),

    #Weeks with medium demand
    (2024, 44),
    (2025, 14),

    #Weeks with lowest demand
    (2024, 27),
    (2024, 36),
}

# Optional per-floor override. Any floor omitted here falls back to
# ALLOWED_ISO_WEEKS above.
ALLOWED_ISO_WEEKS_BY_FLOOR = {
    2: {
        # Weeks with highest demand
        (2025, 6),
        (2025, 8),
        
        #Weeks with medium demand
        (2024, 44),
        (2025, 14),

        #Weeks with lowest demand
        (2024, 27),
        (2024, 28)
    },
    3: {
        # Weeks with highest demand
        (2025, 5),
        (2025, 10),
        
        #Weeks with medium demand
        (2024, 44),
        (2025, 14),

        #Weeks with lowest demand
        (2024, 46),
        (2025, 2)
    },
    7: {
        # Weeks with highest demand
        (2024, 39),
        (2024, 40),
        
        #Weeks with medium demand
        (2024, 44),
        (2025, 14),

        #Weeks with lowest demand
        (2024, 46),
        (2025, 26)
    },
    9: {
        # Weeks with highest demand
        (2024, 43),
        (2025, 6),
        
        #Weeks with medium demand
        (2024, 44),
        (2025, 14),

        #Weeks with lowest demand
        (2025, 19),
        (2025, 21)
    }
}

# ----------------------------
# POLICY SPEC
# ----------------------------

@dataclass(frozen=True)
class PolicySpec:
    policy_name: str
    mode: int
    alpha: Optional[float] = None
    allow_deallocation: bool = False
    allow_reweighting: bool = False
    extra_args: tuple[str, ...] = ()  # any extra CLI args you want to add


POLICIES: list[PolicySpec] = [
    PolicySpec("fleet_manager", mode=0),

    PolicySpec("tp_d", mode=1, alpha=0.0),
    PolicySpec("tp_d", mode=1, alpha=0.2),

    PolicySpec("d_tpts", mode=2, alpha=0.0),
    PolicySpec("d_tpts", mode=2, alpha=0.2),

    PolicySpec("idle_pred", mode=3),

    PolicySpec("base_policy", mode=4),

    PolicySpec("adaptive_rollout", mode=5, allow_deallocation=True, allow_reweighting=True),
    PolicySpec("adaptive_rollout", mode=6, allow_deallocation=True, allow_reweighting=False),
    PolicySpec("adaptive_rollout", mode=7, allow_deallocation=False, allow_reweighting=True),
    PolicySpec("adaptive_rollout", mode=8, allow_deallocation=False, allow_reweighting=False),
]


# ----------------------------
# DATE HELPERS
# ----------------------------

def daterange_iso_filtered(start: dt.date, end: dt.date,
                           allowed_iso_weeks: set[IsoWeek]) -> Iterable[dt.date]:
    d = start
    while d <= end:
        iso_year, iso_week, _ = d.isocalendar()
        if (iso_year, iso_week) in allowed_iso_weeks:
            yield d
        d += dt.timedelta(days=1)


def resolve_iso_weeks_by_floor(floors: Iterable[int],
                               default_iso_weeks: set[IsoWeek],
                               per_floor_iso_weeks: dict[int, set[IsoWeek]]) -> dict[int, set[IsoWeek]]:
    floor_list = [int(floor) for floor in floors]
    unknown_floors = sorted(set(per_floor_iso_weeks) - set(floor_list))
    if unknown_floors:
        raise ValueError(f"Found ISO-week selections for unknown floors: {unknown_floors}")

    resolved: dict[int, set[IsoWeek]] = {}
    for floor in floor_list:
        weeks = per_floor_iso_weeks.get(floor, default_iso_weeks)
        resolved[floor] = set(weeks)
    return resolved


# ----------------------------
# CMD / LOGGING HELPERS
# ----------------------------

def policy_run_tag(p: PolicySpec) -> str:
    """
    Stable tag for filenames.
    Examples:
      fleet_manager
      tp_d_alpha0.1
      sequential_greedy_ropt
    """
    tag = f"{p.policy_name}"
    if p.alpha is not None:
        tag += f"_alpha{str(p.alpha)}"

    if p.allow_deallocation:
        tag += "_ropt"
    else:
        if p.mode in ROLLOUT_MODES:  # Only add nopt tag for rollout variants
            tag += "_nopt"

    if p.allow_reweighting:
        tag += "_rwt"
    else:
        if p.mode in ROLLOUT_MODES:  # Only add norwt tag for rollout variants
            tag += "_norwt"
    
    return tag


def prediction_cache_args() -> list[str]:
    if not PREDICTION_CACHE_PATH:
        return []
    args = [
        "--prediction_cache_path", PREDICTION_CACHE_PATH,
        "--prediction_match_tolerance_minutes", str(PREDICTION_MATCH_TOLERANCE_MINUTES),
        "--prediction_lookahead_minutes", str(PREDICTION_LOOKAHEAD_MINUTES),
        "--prediction_weight_bin_minutes", str(PREDICTION_WEIGHT_BIN_MINUTES),
        "--prediction_cache_patient_filter_mode", PREDICTION_CACHE_PATIENT_FILTER_MODE,
        "--patient_room_stays_file", PATIENT_ROOM_STAYS_FILE,
        "--admissions_discharges_file", ADMISSIONS_DISCHARGES_FILE,
    ]
    if PREDICTION_CACHE_RUN_NAMES:
        args += ["--prediction_cache_run_names", PREDICTION_CACHE_RUN_NAMES]
    return args


def prediction_weight_args() -> list[str]:
    args = []
    optional_args = [
        ("--prediction_weight_alpha_over", PREDICTION_WEIGHT_ALPHA_OVER),
        ("--prediction_weight_alpha_time", PREDICTION_WEIGHT_ALPHA_TIME),
        ("--prediction_weight_beta_over", PREDICTION_WEIGHT_BETA_OVER),
        ("--prediction_weight_beta_time", PREDICTION_WEIGHT_BETA_TIME),
        ("--prediction_weight_lambda_min", PREDICTION_WEIGHT_LAMBDA_MIN),
        ("--prediction_weight_cold_start_weight", PREDICTION_WEIGHT_COLD_START_WEIGHT),
        (
            "--prediction_weight_fallback_min_positive_prediction_windows",
            PREDICTION_WEIGHT_FALLBACK_MIN_POSITIVE_PREDICTION_WINDOWS,
        ),
        (
            "--prediction_weight_patient_credibility_prior",
            PREDICTION_WEIGHT_PATIENT_CREDIBILITY_PRIOR,
        ),
    ]
    for flag, value in optional_args:
        if value is not None:
            args += [flag, str(value)]
    return args


def build_cmd(day: dt.date, floor: int, p: PolicySpec) -> list[str]:
    cmd = BASE + [
        "--mode", str(p.mode),
        "--policy_name", p.policy_name,
        "--year", str(day.year),
        "--month", str(day.month),
        "--day", str(day.day),
        "--floor_number", str(floor),
    ]
    if p.alpha is not None:
        cmd += ["--alpha", str(p.alpha)]
    if p.mode in CACHE_POLICY_MODES:
        cmd += prediction_cache_args()
    if p.mode in ROLLOUT_MODES and INCLUDE_FUTURE_SCHEDULED_REQUESTS:
        cmd += ["--include_future_scheduled_requests"]
    if p.mode in ROLLOUT_MODES:
        cmd += prediction_weight_args()
    if p.extra_args:
        cmd += list(p.extra_args)
    return cmd


def run_one(day: dt.date, floor: int, p: PolicySpec) -> tuple[dt.date, int, PolicySpec, int]:
    tag = policy_run_tag(p)

    # logs under results/policies/logs/<policy_name>/
    log_dir = os.path.join(LOG_ROOT, tag)
    os.makedirs(log_dir, exist_ok=True)

    base_name = f"{tag}_{day.isoformat()}_floor{floor}"
    out_path = os.path.join(log_dir, base_name + ".out")
    err_path = os.path.join(log_dir, base_name + ".err")

    cmd = build_cmd(day, floor, p)
    print(f"Running {base_name}")

    try:
        with open(out_path, "w") as out, open(err_path, "w") as err:
            subprocess.run(cmd, stdout=out, stderr=err, check=True)
        return day, floor, p, 0
    except subprocess.CalledProcessError as e:
        return day, floor, p, e.returncode


# ----------------------------
# MAIN
# ----------------------------

def main():
    allowed_weeks_by_floor = resolve_iso_weeks_by_floor(
        floors=FLOORS,
        default_iso_weeks=ALLOWED_ISO_WEEKS,
        per_floor_iso_weeks=ALLOWED_ISO_WEEKS_BY_FLOOR,
    )

    floor_days: dict[int, list[dt.date]] = {}
    for floor in FLOORS:
        days = list(
            daterange_iso_filtered(
                START,
                END,
                allowed_iso_weeks=allowed_weeks_by_floor[floor],
            )
        )
        floor_days[floor] = days
        weeks_seen = sorted({(d.isocalendar().year, d.isocalendar().week) for d in days})
        print(f"Floor {floor} allowed weeks: {sorted(allowed_weeks_by_floor[floor])}")
        print(f"Floor {floor} weeks to run: {weeks_seen}")
        print(f"Floor {floor} total days: {len(days)}")

    print("Floors:", FLOORS)
    print("Policies:", [policy_run_tag(p) for p in POLICIES])
    if PREDICTION_CACHE_PATH:
        print("Prediction cache for idle_pred/rollout:", PREDICTION_CACHE_PATH)
        print("Prediction cache run names:", PREDICTION_CACHE_RUN_NAMES or "<all>")
        print("Prediction lookahead minutes:", PREDICTION_LOOKAHEAD_MINUTES)
        print("Prediction weight bin minutes:", PREDICTION_WEIGHT_BIN_MINUTES)
        print("Prediction match tolerance minutes:", PREDICTION_MATCH_TOLERANCE_MINUTES)
        print("Prediction cache patient filter mode:", PREDICTION_CACHE_PATIENT_FILTER_MODE)
        print("Patient room stays file:", PATIENT_ROOM_STAYS_FILE)
        print("Admissions/discharges file:", ADMISSIONS_DISCHARGES_FILE)
    else:
        print("Prediction cache disabled for idle_pred/rollout.")

    jobs = [(day, floor, p) for floor in FLOORS for day in floor_days[floor] for p in POLICIES]

    failures: list[tuple[dt.date, int, PolicySpec, int]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_one, d, f, p) for d, f, p in jobs]
        for fut in as_completed(futs):
            day, floor, p, rc = fut.result()
            tag = policy_run_tag(p)
            if rc != 0:
                failures.append((day, floor, p, rc))
                print(f"FAILED {day} floor {floor} {tag} rc={rc}")
            else:
                print(f"OK     {day} floor {floor} {tag}")

    if failures:
        print("\nSome jobs failed:")
        for day, floor, p, rc in failures:
            print(f"  {day} floor {floor} {policy_run_tag(p)} rc={rc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
