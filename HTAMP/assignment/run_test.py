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
END   = dt.date(2024, 7, 7)  # for quick testing; change to 2025-06-29 for full run

MAX_WORKERS = 100  # tune to your machine / cluster limits

BASE = [
    "python", "-m", "HTAMP.evaluate_assignment",
    "--use_saved_data",
    "--use_saved_request_data",
]

LOG_ROOT = "results/policies/logs"


# ----------------------------
# POLICY SPEC
# ----------------------------

@dataclass(frozen=True)
class PolicySpec:
    policy_name: str
    mode: int
    alpha: Optional[float] = None
    allow_deallocation: bool = False
    extra_args: tuple[str, ...] = ()  # for any other flags/params you might need


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
# HELPERS
# ----------------------------

def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def policy_run_tag(p: PolicySpec) -> str:
    """
    Build a stable tag for filenames.
    Examples:
      fleet_manager_m0
      tp_d_m1_a0p1
      sequential_greedy_m4_ropt
    """
    tag = f"{p.policy_name}_m{p.mode}"
    if p.alpha is not None:
        # file-safe alpha: 0.1 -> 0p1
        a = str(p.alpha).replace(".", "p")
        tag += f"_a{a}"
    if p.allow_deallocation:
        tag += "_ropt"
    return tag


def build_cmd(day: dt.date, p: PolicySpec) -> list[str]:
    cmd = BASE + [
        "--mode", str(p.mode),
        "--policy_name", p.policy_name,
        "--year", str(day.year),
        "--month", str(day.month),
        "--day", str(day.day),
    ]
    if p.alpha is not None:
        cmd += ["--alpha", str(p.alpha)]
    if p.allow_deallocation:
        cmd += ["--allow_deallocation"]
    if p.extra_args:
        cmd += list(p.extra_args)
    return cmd


def run_one(day: dt.date, p: PolicySpec) -> tuple[dt.date, PolicySpec, int]:
    tag = policy_run_tag(p)

    # logs go under results/policies/logs/<policy_name>/
    log_dir = os.path.join(LOG_ROOT, p.policy_name)
    os.makedirs(log_dir, exist_ok=True)

    out_path = os.path.join(log_dir, f"{tag}_{day.isoformat()}.out")
    err_path = os.path.join(log_dir, f"{tag}_{day.isoformat()}.err")

    cmd = build_cmd(day, p)
    print(f"Running {tag} for {day} -> {out_path}")

    try:
        with open(out_path, "w") as out, open(err_path, "w") as err:
            subprocess.run(cmd, stdout=out, stderr=err, check=True)
        return day, p, 0
    except subprocess.CalledProcessError as e:
        return day, p, e.returncode


# ----------------------------
# MAIN
# ----------------------------

def main():
    jobs = [(d, p) for d in daterange(START, END) for p in POLICIES]

    failures: list[tuple[dt.date, PolicySpec, int]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_one, d, p) for d, p in jobs]
        for fut in as_completed(futs):
            day, p, rc = fut.result()
            tag = policy_run_tag(p)
            if rc != 0:
                failures.append((day, p, rc))
                print(f"FAILED {day} {tag} rc={rc}")
            else:
                print(f"OK     {day} {tag}")

    if failures:
        print("\nSome jobs failed:")
        for day, p, rc in failures:
            print(f"  {day} {policy_run_tag(p)} rc={rc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()