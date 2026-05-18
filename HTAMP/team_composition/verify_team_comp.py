import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from HTAMP.team_composition.run_team_comp import (
    DISALLOWED_ISO_WEEKS,
    DISALLOWED_ISO_WEEKS_BY_FLOOR,
    END,
    FLOORS,
    MAX_WORKERS,
    START,
    daterange_iso_filtered,
    resolve_iso_weeks_by_floor,
)


DEFAULT_LOG_ROOT = Path("results/team_comp/verification")
TEAM_COMPOSITION_RESULT_PREFIX = "TEAM_COMPOSITION_EVAL_RESULT "
SUMMARY_FIELDS = [
    "timestamp",
    "iteration",
    "monitoring_robots",
    "delivery_robots",
    "total_days",
    "successful_days",
    "success_proportion",
    "required_success_proportion",
    "monitoring_failed_days",
    "delivery_failed_days",
    "unknown_failed_days",
    "log_dir",
]


@dataclass(frozen=True)
class EvaluationJob:
    day: dt.date
    floor: int


@dataclass(frozen=True)
class EvaluationResult:
    day: dt.date
    floor: int
    status: Optional[str]
    reject_type: Optional[str]
    rejection_category: Optional[str]
    return_code: int
    out_path: Path
    err_path: Path
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.status == "success"


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value!r}") from exc


def normalize_required_success(value: str) -> float:
    try:
        numeric_value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a numeric success threshold, got {value!r}") from exc

    if numeric_value < 0:
        raise argparse.ArgumentTypeError("Required success proportion must be non-negative.")
    threshold = numeric_value / 100.0 if numeric_value > 1.0 else numeric_value
    if threshold > 1.0:
        raise argparse.ArgumentTypeError("Required success proportion must be in [0, 1] or percentage in [0, 100].")
    return threshold


def validate_args(args: argparse.Namespace) -> None:
    if args.num_monitoring_robots < 0 or args.num_delivery_robots < 0:
        raise SystemExit("Robot counts must be non-negative.")
    if args.num_monitoring_robots + args.num_delivery_robots == 0:
        raise SystemExit("At least one robot is required.")
    if args.max_workers < 1:
        raise SystemExit("--max_workers must be at least 1.")
    if args.max_iterations < 1:
        raise SystemExit("--max_iterations must be at least 1.")
    if args.start_date > args.end_date:
        raise SystemExit("--start_date must be on or before --end_date.")


def build_training_jobs(args: argparse.Namespace) -> list[EvaluationJob]:
    per_floor_weeks = {
        floor: weeks
        for floor, weeks in DISALLOWED_ISO_WEEKS_BY_FLOOR.items()
        if floor in args.floors
    }
    disallowed_weeks_by_floor = resolve_iso_weeks_by_floor(
        floors=args.floors,
        default_iso_weeks=DISALLOWED_ISO_WEEKS,
        per_floor_iso_weeks=per_floor_weeks,
    )

    jobs: list[EvaluationJob] = []
    for floor in args.floors:
        floor_days = list(
            daterange_iso_filtered(
                args.start_date,
                args.end_date,
                disallowed_iso_weeks=disallowed_weeks_by_floor[floor],
            )
        )
        weeks_seen = sorted({(d.isocalendar().year, d.isocalendar().week) for d in floor_days})
        print(f"Floor {floor} disallowed weeks: {sorted(disallowed_weeks_by_floor[floor])}")
        print(f"Floor {floor} weeks to evaluate: {weeks_seen}")
        print(f"Floor {floor} training days: {len(floor_days)}")
        jobs.extend(EvaluationJob(day=day, floor=floor) for day in floor_days)

    if not jobs:
        raise SystemExit("No training days were selected.")

    return jobs


def add_common_stability_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend([
        "--hour_start", str(args.hour_start),
        "--hour_end", str(args.hour_end),
        "--request_dir", args.request_dir,
        "--medications_orders_file", args.medications_orders_file,
        "--blood_pressure_orders_file", args.blood_pressure_orders_file,
        "--heart_rate_orders_file", args.heart_rate_orders_file,
        "--respiratory_rate_orders_file", args.respiratory_rate_orders_file,
        "--temperature_orders_file", args.temperature_orders_file,
        "--oxygen_saturation_orders_file", args.oxygen_saturation_orders_file,
        "--config_path", args.config_path,
        "--occupancy_map_path", args.occupancy_map_path,
        "--factor", str(args.factor),
        "--meters_per_pixel", str(args.meters_per_pixel),
        "--fps", str(args.fps),
        "--occupancy_reservations_file", args.occupancy_reservations_file,
        "--random_seed", str(args.random_seed),
    ])
    if not args.recompute_occupancy_data:
        cmd.append("--use_saved_data")
    if not args.recompute_request_data:
        cmd.append("--use_saved_request_data")


def build_eval_cmd(args: argparse.Namespace,
                   job: EvaluationJob,
                   num_monitoring_robots: int,
                   num_delivery_robots: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m", "HTAMP.team_composition.stability_eval",
        "--evaluate_fixed_team",
        "--year", str(job.day.year),
        "--month", str(job.day.month),
        "--day", str(job.day.day),
        "--floor_number", str(job.floor),
        "--num_monitoring_robots", str(num_monitoring_robots),
        "--num_delivery_robots", str(num_delivery_robots),
    ]
    add_common_stability_args(cmd, args)
    return cmd


def parse_result_marker(out_path: Path) -> tuple[Optional[dict], Optional[str]]:
    try:
        lines = out_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return None, f"Could not read output log: {exc}"

    for line in reversed(lines):
        if line.startswith(TEAM_COMPOSITION_RESULT_PREFIX):
            payload = line[len(TEAM_COMPOSITION_RESULT_PREFIX):].strip()
            try:
                return json.loads(payload), None
            except json.JSONDecodeError as exc:
                return None, f"Could not parse result marker JSON: {exc}"

    return None, "Missing team-composition result marker."


def run_one(args: argparse.Namespace,
            job: EvaluationJob,
            num_monitoring_robots: int,
            num_delivery_robots: int,
            log_dir: Path) -> EvaluationResult:
    base_name = f"teamcomp_{job.day.isoformat()}_floor{job.floor}"
    out_path = log_dir / f"{base_name}.out"
    err_path = log_dir / f"{base_name}.err"
    cmd = build_eval_cmd(args, job, num_monitoring_robots, num_delivery_robots)

    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        completed = subprocess.run(cmd, stdout=out, stderr=err, check=False)

    payload, parse_error = parse_result_marker(out_path)
    error_parts = []
    if completed.returncode != 0:
        error_parts.append(f"process exited with return code {completed.returncode}")
    if parse_error is not None:
        error_parts.append(parse_error)

    status = payload.get("status") if payload else None
    if payload and status not in {"success", "rejected"}:
        error_parts.append(f"unexpected result status {status!r}")

    return EvaluationResult(
        day=job.day,
        floor=job.floor,
        status=status,
        reject_type=payload.get("reject_type") if payload else None,
        rejection_category=payload.get("rejection_category") if payload else None,
        return_code=completed.returncode,
        out_path=out_path,
        err_path=err_path,
        error="; ".join(error_parts) if error_parts else None,
    )


def evaluate_composition(args: argparse.Namespace,
                         jobs: list[EvaluationJob],
                         num_monitoring_robots: int,
                         num_delivery_robots: int,
                         iteration: int) -> list[EvaluationResult]:
    log_dir = Path(args.log_root) / f"iteration_{iteration:03d}_monitoring{num_monitoring_robots}_delivery{num_delivery_robots}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print("")
    print(f"Iteration {iteration}: evaluating {num_monitoring_robots} monitoring, {num_delivery_robots} delivery across {len(jobs)} training days.")
    print(f"Logs: {log_dir}")

    results: list[EvaluationResult] = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as executor:
        futures = [
            executor.submit(run_one, args, job, num_monitoring_robots, num_delivery_robots, log_dir)
            for job in jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.error is not None:
                print(f"ERROR    {result.day} floor {result.floor}: {result.error}")
            elif result.success:
                print(f"OK       {result.day} floor {result.floor}")
            else:
                print(f"REJECTED {result.day} floor {result.floor}: {result.rejection_category} ({result.reject_type})")

    technical_errors = [result for result in results if result.error is not None]
    if technical_errors:
        print("")
        print("Some evaluations failed before producing a valid result marker:")
        for result in technical_errors[:20]:
            print(f"  {result.day} floor {result.floor}: {result.error}")
            print(f"    stdout: {result.out_path}")
            print(f"    stderr: {result.err_path}")
        if len(technical_errors) > 20:
            print(f"  ...and {len(technical_errors) - 20} more.")
        raise SystemExit(1)

    return results


def summarize_results(results: list[EvaluationResult], required_success_proportion: float) -> dict[str, object]:
    total_days = len(results)
    successful_days = sum(1 for result in results if result.success)
    monitoring_failed_days = sum(1 for result in results if result.rejection_category == "monitoring")
    delivery_failed_days = sum(1 for result in results if result.rejection_category == "delivery")
    unknown_failed_days = total_days - successful_days - monitoring_failed_days - delivery_failed_days
    return {
        "total_days": total_days,
        "successful_days": successful_days,
        "success_proportion": successful_days / total_days,
        "required_success_proportion": required_success_proportion,
        "monitoring_failed_days": monitoring_failed_days,
        "delivery_failed_days": delivery_failed_days,
        "unknown_failed_days": unknown_failed_days,
    }


def append_iteration_summary(args: argparse.Namespace,
                             iteration: int,
                             num_monitoring_robots: int,
                             num_delivery_robots: int,
                             summary: dict[str, object],
                             log_dir: Path) -> None:
    summary_path = Path(args.log_root) / "team_composition_search.csv"
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "monitoring_robots": num_monitoring_robots,
        "delivery_robots": num_delivery_robots,
        "log_dir": str(log_dir),
        **summary,
    }
    write_header = not summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def choose_robot_type_to_increment(args: argparse.Namespace, summary: dict[str, object]) -> str:
    monitoring_failed_days = int(summary["monitoring_failed_days"])
    delivery_failed_days = int(summary["delivery_failed_days"])
    if delivery_failed_days > monitoring_failed_days:
        return "delivery"
    if monitoring_failed_days > delivery_failed_days:
        return "monitoring"
    if monitoring_failed_days == 0 and delivery_failed_days == 0:
        raise SystemExit("Success proportion was too low, but no rejected training days were counted.")
    return args.tie_breaker


def write_final_summary(args: argparse.Namespace,
                        iteration: int,
                        num_monitoring_robots: int,
                        num_delivery_robots: int,
                        summary: dict[str, object],
                        initial_monitoring_robots: int,
                        initial_delivery_robots: int) -> None:
    final_data = {
        "iteration": iteration,
        "initial_monitoring_robots": initial_monitoring_robots,
        "initial_delivery_robots": initial_delivery_robots,
        "final_monitoring_robots": num_monitoring_robots,
        "final_delivery_robots": num_delivery_robots,
        **summary,
    }
    log_root = Path(args.log_root)
    (log_root / "final_team_composition.json").write_text(
        json.dumps(final_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (log_root / "final_team_composition.txt").write_text(
        "\n".join([
            f"Monitoring Robots: {num_monitoring_robots}",
            f"Delivery Robots: {num_delivery_robots}",
            f"Successful Days: {summary['successful_days']} / {summary['total_days']}",
            f"Success Proportion: {float(summary['success_proportion']):.6f}",
            f"Required Success Proportion: {float(summary['required_success_proportion']):.6f}",
        ]) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify_team_comp.py",
        description="Verify and grow a user-specified team composition over the training dates used by run_team_comp.py.",
    )
    parser.add_argument("--num_monitoring_robots", "--monitoring_robots", type=int, required=True, help="Initial number of monitoring robots.")
    parser.add_argument("--num_delivery_robots", "--delivery_robots", type=int, required=True, help="Initial number of delivery robots.")
    parser.add_argument("--required_success_proportion", "--required_percentage", type=normalize_required_success, required=True, help="Required serviced-day threshold. Values in [0, 1] are proportions; values above 1 are interpreted as percentages.")
    parser.add_argument("--tie_breaker", choices=["monitoring", "delivery"], default="monitoring", help="Robot type to increment when monitoring and delivery have the same number of failed days.")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum number of team-composition growth iterations.")

    parser.add_argument("--start_date", type=parse_date, default=START, help="First date to evaluate, inclusive.")
    parser.add_argument("--end_date", type=parse_date, default=END, help="Last date to evaluate, inclusive.")
    parser.add_argument("--floors", type=int, nargs="+", default=FLOORS, help="Floor numbers to evaluate.")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS, help="Maximum parallel day/floor evaluations.")
    parser.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT, help="Directory for per-day logs and search summaries.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed used to select robot starting positions.")

    parser.add_argument("--hour_start", type=int, default=0, help="Starting hour of operational range.")
    parser.add_argument("--hour_end", type=int, default=24, help="Ending hour of operational range.")
    parser.add_argument("--request_dir", type=str, default="data/requests", help="Directory containing saved global request data.")
    parser.add_argument("--recompute_occupancy_data", action="store_true", help="Recompute occupancy reservations instead of passing --use_saved_data.")
    parser.add_argument("--recompute_request_data", action="store_true", help="Recompute request data instead of passing --use_saved_request_data.")

    parser.add_argument("--medications_orders_file", type=str, default="data/processed/medication_orders_annotated.csv", help="Path to the medications orders CSV file.")
    parser.add_argument("--blood_pressure_orders_file", type=str, default="data/processed/blood_pressure_orders_annotated.csv", help="Path to the blood pressure orders CSV file.")
    parser.add_argument("--heart_rate_orders_file", type=str, default="data/processed/heart_rate_orders_annotated.csv", help="Path to the heart rate orders CSV file.")
    parser.add_argument("--respiratory_rate_orders_file", type=str, default="data/processed/respiratory_rate_orders_annotated.csv", help="Path to the respiratory rate orders CSV file.")
    parser.add_argument("--temperature_orders_file", type=str, default="data/processed/temperature_orders_annotated.csv", help="Path to the temperature orders CSV file.")
    parser.add_argument("--oxygen_saturation_orders_file", type=str, default="data/processed/oxygen_saturation_orders_annotated.csv", help="Path to the oxygen saturation orders CSV file.")

    parser.add_argument("--config_path", type=str, default="maps/hospital_floor/floor_config.yaml", help="Path to the configuration file.")
    parser.add_argument("--occupancy_map_path", type=str, default="maps/hospital_floor/occupancy_map.npy", help="Path to the input occupancy map.")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor.")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image.")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second for the grid world.")
    parser.add_argument("--occupancy_reservations_file", type=str, default="data/occupancy_reservations.pkl", help="Path to the occupancy reservations file.")

    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    Path(args.log_root).mkdir(parents=True, exist_ok=True)

    initial_monitoring_robots = args.num_monitoring_robots
    initial_delivery_robots = args.num_delivery_robots
    num_monitoring_robots = initial_monitoring_robots
    num_delivery_robots = initial_delivery_robots
    jobs = build_training_jobs(args)

    print("")
    print(f"Required success proportion: {args.required_success_proportion:.6f}")
    print(f"Total training floor-days: {len(jobs)}")

    for iteration in range(1, args.max_iterations + 1):
        results = evaluate_composition(
            args=args,
            jobs=jobs,
            num_monitoring_robots=num_monitoring_robots,
            num_delivery_robots=num_delivery_robots,
            iteration=iteration,
        )
        summary = summarize_results(results, args.required_success_proportion)
        log_dir = Path(args.log_root) / f"iteration_{iteration:03d}_monitoring{num_monitoring_robots}_delivery{num_delivery_robots}"
        append_iteration_summary(
            args=args,
            iteration=iteration,
            num_monitoring_robots=num_monitoring_robots,
            num_delivery_robots=num_delivery_robots,
            summary=summary,
            log_dir=log_dir,
        )

        print("")
        print(f"Successful training days: {summary['successful_days']} / {summary['total_days']} ({float(summary['success_proportion']):.6f})")
        print(f"Monitoring failed days: {summary['monitoring_failed_days']}")
        print(f"Delivery failed days: {summary['delivery_failed_days']}")
        print(f"Unknown failed days: {summary['unknown_failed_days']}")

        if float(summary["success_proportion"]) >= args.required_success_proportion:
            write_final_summary(
                args=args,
                iteration=iteration,
                num_monitoring_robots=num_monitoring_robots,
                num_delivery_robots=num_delivery_robots,
                summary=summary,
                initial_monitoring_robots=initial_monitoring_robots,
                initial_delivery_robots=initial_delivery_robots,
            )
            if iteration == 1:
                print("The original team composition is sufficient.")
            print("Final team composition:")
            print(f"Monitoring Robots: {num_monitoring_robots}, Delivery Robots: {num_delivery_robots}")
            print(f"Summary log: {Path(args.log_root) / 'team_composition_search.csv'}")
            print(f"Final log: {Path(args.log_root) / 'final_team_composition.txt'}")
            return

        robot_type_to_increment = choose_robot_type_to_increment(args, summary)
        if robot_type_to_increment == "delivery":
            num_delivery_robots += 1
        else:
            num_monitoring_robots += 1
        print(f"Increasing {robot_type_to_increment} robots by 1.")

    raise SystemExit(f"Reached --max_iterations={args.max_iterations} before satisfying the required success proportion.")


if __name__ == "__main__":
    main()
