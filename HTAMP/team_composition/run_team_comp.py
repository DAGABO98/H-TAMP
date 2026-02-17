import datetime as dt
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

START = dt.date(2024, 6, 24)
END   = dt.date(2025, 6, 29)
FLOORS = [2, 3, 7, 9]
MAX_WORKERS = 100

BASE = [
    "python", "-m", "HTAMP.team_composition.stability_eval",
    "--use_saved_data",
    "--use_saved_request_data",
    "--hour_start", "8",
    "--hour_end", "9",
]

def run_one(day: dt.date, floor: int) -> tuple[dt.date, int, int]:
    print(f"Running team composition for {day} floor {floor}")
    cmd = BASE + [
        "--year", str(day.year),
        "--month", str(day.month),
        "--day", str(day.day),
        "--floor_number", str(floor),
    ]

    os.makedirs("results/team_comp/logs", exist_ok=True)
    out_path = f"results/team_comp/logs/teamcomp_{day.isoformat()}_floor{floor}.out"
    err_path = f"results/team_comp/logs/teamcomp_{day.isoformat()}_floor{floor}.err"

    try:
        with open(out_path, "w") as out, open(err_path, "w") as err:
            subprocess.run(cmd, stdout=out, stderr=err, check=True)
        return day, floor, 0
    except subprocess.CalledProcessError as e:
        return day, floor, e.returncode

def main():
    jobs = [(d, f) for d in daterange(START, END) for f in FLOORS]

    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_one, d, f) for d, f in jobs]
        for fut in as_completed(futs):
            day, floor, rc = fut.result()
            if rc != 0:
                failures.append((day, floor, rc))
                print(f"FAILED {day} floor {floor} rc={rc}")
            else:
                print(f"OK     {day} floor {floor}")

    if failures:
        print("\nSome jobs failed:")
        for day, floor, rc in failures:
            print(f"  {day} floor {floor} rc={rc}")
        raise SystemExit(1)

def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)

if __name__ == "__main__":
    main()
