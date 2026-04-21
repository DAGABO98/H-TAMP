import datetime as dt
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Iterable

IsoWeek = tuple[int, int]

START = dt.date(2024, 6, 24)
END   = dt.date(2025, 6, 29)
FLOORS = [2, 3, 7, 9]
MAX_WORKERS = 100

DISALLOWED_ISO_WEEKS: set[IsoWeek] = {
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
# DISALLOWED_ISO_WEEKS above.
DISALLOWED_ISO_WEEKS_BY_FLOOR = {
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

BASE = [
    "python", "-m", "HTAMP.team_composition.stability_eval",
    "--use_saved_data",
    "--use_saved_request_data",
    "--hour_start", "0",
    "--hour_end", "24",
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

def main():
    disallowed_weeks_by_floor = resolve_iso_weeks_by_floor(
        floors=FLOORS,
        default_iso_weeks=DISALLOWED_ISO_WEEKS,
        per_floor_iso_weeks=DISALLOWED_ISO_WEEKS_BY_FLOOR,
    )

    jobs: list[tuple[dt.date, int]] = []
    for floor in FLOORS:
        floor_days = list(
            daterange_iso_filtered(
                START,
                END,
                disallowed_iso_weeks=disallowed_weeks_by_floor[floor],
            )
        )
        weeks_seen = sorted({(d.isocalendar().year, d.isocalendar().week) for d in floor_days})
        print(f"Floor {floor} disallowed weeks: {sorted(disallowed_weeks_by_floor[floor])}")
        print(f"Floor {floor} weeks to run: {weeks_seen}")
        print(f"Floor {floor} total days: {len(floor_days)}")
        jobs.extend((day, floor) for day in floor_days)

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

def daterange_iso_filtered(start: dt.date, end: dt.date,
                           disallowed_iso_weeks: set[IsoWeek]) -> Iterable[dt.date]:
    d = start
    while d <= end:
        iso_year, iso_week, _ = d.isocalendar()
        if (iso_year, iso_week) not in disallowed_iso_weeks:
            yield d
        d += dt.timedelta(days=1)

if __name__ == "__main__":
    main()
