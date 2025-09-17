import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Any
from matplotlib.patches import Rectangle, Circle

from HTAMP.grid_world import Interval

class ReservationsUtils:
    def merge_intervals(intervals: List[Interval]) -> List[Interval]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            ls, le = merged[-1]
            if s <= le:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        return merged

    def complement_intervals(blocked: List[Interval], horizon: float = INF) -> List[Interval]:
        """Return safe intervals in [0, horizon) given blocked intervals (merged)."""
        blocked = merge_intervals([(max(0.0, s), e if e != INF else horizon) for s, e in blocked])
        safe: List[Interval] = []
        t = 0.0
        for s, e in blocked:
            if t < s:
                safe.append((t, s))
            t = max(t, e)
            if t >= horizon:
                break
        if t < horizon:
            safe.append((t, horizon))
        return safe

    def interval_intersection(a: Interval, b: Interval) -> Optional[Interval]:
        s = max(a[0], b[0])
        e = min(a[1], b[1])
        return (s, e) if s < e else None

