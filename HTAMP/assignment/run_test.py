import datetime as dt
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dataclasses import dataclass
from typing import Optional, Iterable


# ----------------------------
# CONFIG
# ----------------------------

START = dt.date(2024, 6, 24)
END   = dt.date(2025, 6, 29)
FLOORS = [2, 3, 7, 9]
MAX_WORKERS = 32  # tune for your machine / cluster

BASE = [
    "python", "-m", "HTAMP.evaluate_assignment",
    "--use_saved_data",
    "--use_saved_request_data",
    "--hour_start", "8",
    "--hour_end", "9",
]

LOG_ROOT = "results/policies/logs"


# Only include dates whose (ISO year, ISO week) is in this explicit list
ALLOWED_ISO_WEEKS: set[tuple[int, int]] = {
    # Weeks with highest demand
    (2024, 40),
    (2025, 5),

    #Weeks with medium demand
    (2024, 44),
    (2025, 14),

    #Weeks with lowest demand
    (2024, 48),
    (2024, 36),
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
    extra_args: tuple[str, ...] = ()  # any extra CLI args you want to add


POLICIES: list[PolicySpec] = [
    PolicySpec("fleet_manager", mode=0),

    PolicySpec("tp_d", mode=1, alpha=0.0),
    PolicySpec("tp_d", mode=1, alpha=0.1),
    PolicySpec("tp_d", mode=1, alpha=0.2),

    PolicySpec("d_tpts", mode=2, alpha=0.0),
    PolicySpec("d_tpts", mode=2, alpha=0.1),
    PolicySpec("d_tpts", mode=2, alpha=0.2),

    PolicySpec("sequential_greedy", mode=4),
    PolicySpec("sequential_greedy", mode=4, allow_deallocation=True),
]


# ----------------------------
# DATE HELPERS
# ----------------------------

def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def daterange_iso_filtered(start: dt.date, end: dt.date,
                           allowed_iso_weeks: set[tuple[int, int]]) -> Iterable[dt.date]:
    d = start
    while d <= end:
        iso_year, iso_week, _ = d.isocalendar()
        if (iso_year, iso_week) in allowed_iso_weeks:
            yield d
        d += dt.timedelta(days=1)


# ----------------------------
# CMD / LOGGING HELPERS
# ----------------------------

def policy_run_tag(p: PolicySpec) -> str:
    """
    Stable tag for filenames.
    Examples:
      fleet_manager_m0
      tp_d_m1_a0p1
      sequential_greedy_m4_ropt
    """
    tag = f"{p.policy_name}_m{p.mode}"
    if p.alpha is not None:
        tag += f"_a{str(p.alpha).replace('.', 'p')}"
    if p.allow_deallocation:
        tag += "_ropt"
    return tag


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
    if p.allow_deallocation:
        cmd += ["--allow_deallocation"]
    if p.extra_args:
        cmd += list(p.extra_args)
    return cmd


def run_one(day: dt.date, floor: int, p: PolicySpec) -> tuple[dt.date, PolicySpec, int]:
    tag = policy_run_tag(p)

    # logs under results/policies/logs/<policy_name>/
    log_dir = os.path.join(LOG_ROOT, p.policy_name)
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
    days = list(daterange_iso_filtered(START, END, ALLOWED_ISO_WEEKS))

    # sanity check: ensure we only run those weeks
    weeks_seen = sorted({(d.isocalendar().year, d.isocalendar().week) for d in days})
    print("Weeks to run:", weeks_seen)
    print("Total days:", len(days))
    print("Floors:", FLOORS)
    print("Policies:", [policy_run_tag(p) for p in POLICIES])

    jobs = [(d, f, p) for d in days for f in FLOORS for p in POLICIES]

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