import heapq
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any
from matplotlib.patches import Rectangle, Circle

from HTAMP.grid_world import Coordinate, TimeInterval, GridWorld, GridIndex, RobotProfile

@dataclass
class TimeReservation:
    interval: TimeInterval
    robot_id: int

    def overlaps(self, other: "TimeReservation") -> bool:
        return self.interval.start < other.interval.end and self.interval.end > other.interval.start
    
    def merge(self, other: "TimeReservation") -> "TimeReservation":
        if not self.overlaps(other):
            raise ValueError("Intervals do not overlap and cannot be merged.")
        
        if self.robot_id != other.robot_id:
            raise ValueError("Cannot merge reservations for different robots.")
        
        new_start = min(self.interval.start, other.interval.start)
        new_end = max(self.interval.end, other.interval.end)
        return TimeReservation(TimeInterval(new_start, new_end), self.robot_id)
    
    def duration(self) -> float:
        return self.interval.end - self.interval.start

    def complement(self, horizon: float) -> List["TimeReservation"]:
        """Return the complement intervals within [0, horizon)."""
        complements: List[TimeReservation] = []
        if self.interval.start > 0:
            complements.append(TimeReservation(TimeInterval(0, self.interval.start), self.robot_id))
        if self.interval.end < horizon:
            complements.append(TimeReservation(TimeInterval(self.interval.end, horizon), self.robot_id))
        return complements
    
    def __repr__(self) -> str:
        return f"TimeReservation(start={self.interval.start}, end={self.interval.end}, robot_id={self.robot_id})"

@dataclass
class ReservationTable:
    reservations: Dict[GridIndex, List[TimeReservation]]
    robot_cell_dict: Dict[int, List[GridIndex]] = field(default_factory=dict)

    def get_reservations(self, cell: GridIndex) -> List[TimeReservation]:
        return self.reservations.get(cell, [])

    def add_reservation(self, cell: GridIndex, reservation: TimeReservation) -> None:
        if cell in self.reservations:
            for i, existing in enumerate(self.reservations[cell]):
                if existing.overlaps(reservation) and existing.robot_id == reservation.robot_id:
                    merged = existing.merge(reservation)
                    self.reservations[cell][i] = merged
                    return merged
            self.reservations[cell].append(reservation)
            self.robot_cell_dict.setdefault(reservation.robot_id, []).append(cell)
        else:
            self.reservations[cell] = [reservation]
            self.robot_cell_dict.setdefault(reservation.robot_id, []).append(cell)
        return reservation

    def _remove_cell_reservation_for_robot(self, cell: GridIndex, robot_id: int) -> None:
        if cell in self.reservations:
            self.reservations[cell] = [r for r in self.reservations[cell] if r.robot_id != robot_id]
            if not self.reservations[cell]:
                del self.reservations[cell]
    
    def remove_reservations_for_robot(self, robot_id: int) -> None:
        if robot_id in self.robot_cell_dict:
            for cell in self.robot_cell_dict[robot_id]:
                self._remove_cell_reservation_for_robot(cell, robot_id)
            del self.robot_cell_dict[robot_id]

    def check_conflict(self, cell: GridIndex, reservation: TimeReservation) -> bool:
        for existing in self.get_reservations(cell):
            if existing.overlaps(reservation) and existing.robot_id != reservation.robot_id:
                return True
        return False

    def get_safe_intervals(self, 
                           cell: GridIndex,
                           horizon: float = float('inf'), 
                           robot_id: int = 1) -> List[TimeReservation]:
        blocked = sorted(self.get_reservations(cell), key=lambda r: r.interval.start)
        safe_intervals: List[TimeReservation] = []
        current_time = 0.0
        for reservation in blocked:
            if reservation.interval.start > current_time:
                safe_intervals.append(TimeReservation(TimeInterval(current_time, reservation.interval.start), robot_id=robot_id))
            current_time = max(current_time, reservation.interval.end)
        if current_time < horizon:
            safe_intervals.append(TimeReservation(TimeInterval(current_time, horizon), robot_id=robot_id))
        return safe_intervals

@dataclass(order=True)
class PQItem:
    f: int
    g: int
    node: "SIPPNode" = field(compare=False)

@dataclass
class SIPPNode:
    pos: Coordinate
    interval: TimeInterval
    arrival: int
    parent: Optional["SIPPNode"] = None

class SIPPwRT:
    def __init__(self, 
                 grid: GridWorld, 
                 reservation_table: Optional[ReservationTable]):
        self.grid = grid
        self.reservation_table = reservation_table
    
    def heuristic(self, 
                  pos: Coordinate, 
                  goal: Coordinate) -> float:
        return np.linalg.norm(np.array([pos.x - goal.x, pos.y - goal.y]))
    
    def _intersect_intervals(self, 
                             interval_list1: List[TimeReservation],
                             interval_list2: List[TimeReservation]) -> List[TimeReservation]:
        result: List[TimeReservation] = []
        i, j = 0, 0
        while i < len(interval_list1) and j < len(interval_list2):
            a, b = interval_list1[i], interval_list2[j]
            if a.interval.end <= b.interval.start:
                i += 1
            elif b.interval.end <= a.interval.start:
                j += 1
            else:
                start = max(a.interval.start, b.interval.start)
                end = min(a.interval.end, b.interval.end)
                if start < end:
                    result.append(TimeReservation(TimeInterval(start, end), robot_id=-1))
                if a.interval.end < b.interval.end:
                    i += 1
                else:
                    j += 1
        return result

    def _get_safe_intervals_for_move(self, 
                                     start_pos: Coordinate, 
                                     end_pos: Coordinate,
                                     robot_profile: RobotProfile,
                                     horizon: float = float('inf')) -> List[TimeReservation]:
        safe_intervals: List[TimeReservation] = []
        robot_occupancy_list = self.grid.get_robot_occupancy_for_move(robot_start_pos=start_pos,
                                                                 robot_end_pos=end_pos,
                                                                 robot_profile=robot_profile)
        occupied_cells: Set[GridIndex] = set()
        for robot_occupancy in robot_occupancy_list:
            occupied_cells.update(robot_occupancy.occupied_cells)

        for cell in occupied_cells:
            cell_safe_intervals = self.reservation_table.get_safe_intervals(cell, 
                                                                            horizon=horizon, 
                                                                            robot_id=robot_profile.robot_id)
            if not safe_intervals:
                safe_intervals = cell_safe_intervals
            else:
                safe_intervals = self._intersect_intervals(safe_intervals, 
                                                           cell_safe_intervals)

        return safe_intervals
    
    def check_conflict_for_move(self,
                                start_pos: Coordinate, 
                                end_pos: Coordinate,
                                robot_profile: RobotProfile, 
                                current_time: float) -> bool:
        # Check if moving from start_pos to end_pos at current_time causes a conflict
        end_time = current_time + self._get_travel_time(start_pos, end_pos, robot_profile)
        robot_reservations = self.grid.get_robot_reservations_for_move(robot_start_pos=start_pos,
                                                                       robot_end_pos=end_pos,
                                                                       robot_profile=robot_profile,
                                                                       current_time=current_time,
                                                                       end_time=end_time)
        
        for robot_reservation in robot_reservations:
            for cell in robot_reservation.robot_occupancy.occupied_cells:
                time_reservation = TimeReservation(interval=robot_reservation.time_interval, 
                                                   robot_id=robot_profile.robot_id)
                if self.reservation_table.check_conflict(cell, time_reservation):
                    return True
        return False


    def _get_travel_time(self, 
                         start_pos: Coordinate, 
                         end_pos: Coordinate, 
                         robot_profile: RobotProfile) -> float:
        dev_x = end_pos.x - start_pos.x
        dev_y = end_pos.y - start_pos.y
        total_displacement = np.sqrt(dev_x**2 + dev_y**2)
        travel_time = total_displacement / robot_profile.velocity if robot_profile.velocity > 0 else 0
        return travel_time

    def _get_earliest_departure(self,
                                curr_node: SIPPNode,
                                start_pos: Coordinate,
                                end_pos: Coordinate,
                                robot_profile: RobotProfile,
                                end_pos_interval: TimeInterval) -> Optional[float]:
        # Find the earliest departure time for the current node
        current_time = max(curr_node.arrival, curr_node.interval.start)
        travel_time = self._get_travel_time(start_pos, end_pos, robot_profile)

        window_start = end_pos_interval.start - travel_time
        current_time = max(current_time, window_start)

        if self.check_conflict_for_move(start_pos=start_pos, 
                                        end_pos=end_pos, 
                                        robot_profile=robot_profile,
                                        current_time=current_time,
                                        travel_time=travel_time):
            return None
        if current_time + travel_time <= end_pos_interval.end:
            return current_time
        return None

    def _check_goal(self, 
                    pos: Coordinate, 
                    goal: Coordinate, 
                    robot_profile: RobotProfile) -> bool:
        return np.linalg.norm(np.array([pos.x - goal.x, pos.y - goal.y])) <= robot_profile.radius

    def _reconstruct_path(self, 
                          node: SIPPNode, 
                          robot_profile: RobotProfile) -> List[Tuple[Coordinate, TimeInterval]]:
        node_list : List[SIPPNode] = []
        while node:
            node_list.append(node)
            node = node.parent
        
        node_list.reverse()
        timed_path: List[Tuple[Coordinate, TimeInterval]] = []
        for i in range(len(node_list) - 1):
            n = node_list[i]
            next_n = node_list[i + 1]
            travel_time = self._get_travel_time(n.pos, next_n.pos, robot_profile)
            departure_time = next_n.arrival - travel_time
            timed_path.append((n.pos, TimeInterval(n.arrival, departure_time)))
        
        # Add the last node with an open-ended time interval
        last_node = node_list[-1]
        timed_path.append((last_node.pos, TimeInterval(last_node.arrival, float('inf'))))

        return timed_path

    def plan_path(self, 
                  start_pos: Coordinate, 
                  goal_pos: Coordinate, 
                  robot_profile: RobotProfile,
                  current_time: float = 0.0,
                  horizon: float = float('inf')) -> Optional[List[Tuple[Coordinate, TimeInterval]]]:
        # Plan the path from start to goal while avoiding obstacles
        start_safe_intervals = self._get_safe_intervals_for_move(start_pos,
                                                                 start_pos,
                                                                 robot_profile,
                                                                 horizon)
        assert start_safe_intervals, "No safe intervals at start position"

        node_list: List[SIPPNode] = []
        for time_reservation in start_safe_intervals:
            if time_reservation.interval.end < current_time:
                continue
            arrival_time = max(time_reservation.interval.start, current_time)
            node_list.append(SIPPNode(pos=start_pos, interval=time_reservation.interval, arrival=arrival_time))
        
        if not node_list:
            return None
        
        open_set: List[PQItem] = []
        seen_set: Dict[Tuple[Coordinate, TimeInterval], float] = {}
        for node in node_list:
            f = node.arrival + self.heuristic(node.pos, goal_pos)
            heapq.heappush(open_set, PQItem(f=f, g=node.arrival, node=node))
            seen_set[(node.pos, node.interval)] = node.arrival

        while open_set:
            current_item = heapq.heappop(open_set)
            current_node = current_item.node

            if self._check_goal(pos=current_node.pos, goal=goal_pos, robot_profile=robot_profile):
                return self._reconstruct_path(current_node)

            for potential_next_move in self.grid.get_valid_moves(robot_center=current_node.pos,
                                                                 robot_profile=robot_profile):
                for safe_interval in self._get_safe_intervals_for_move(start_pos=current_node.pos,
                                                                      end_pos=potential_next_move,
                                                                      robot_profile=robot_profile,
                                                                      horizon=horizon):
                    
                    earliest_departure = self._get_earliest_departure(curr_node=current_node,
                                                                      start_pos=current_node.pos,
                                                                      end_pos=potential_next_move,
                                                                      robot_profile=robot_profile,
                                                                      end_pos_interval=safe_interval.interval)
                    if earliest_departure is None: 
                        continue

                    travel_time = self._get_travel_time(current_node.pos, potential_next_move, robot_profile)
                    arrival_time = earliest_departure + travel_time
                    child_node = SIPPNode(pos=potential_next_move,
                                          interval=safe_interval.interval,
                                          arrival=arrival_time,
                                          parent=current_node)
                    child_key = (child_node.pos, child_node.interval)
                    g_prev = seen_set.get(child_key)
                    if g_prev is None or arrival_time < g_prev:
                        seen_set[child_key] = arrival_time
                        f = arrival_time + self.heuristic(child_node.pos, goal_pos)
                        heapq.heappush(open_set, PQItem(f=f, g=arrival_time, node=child_node))

        return None

class MotionPlanner:
    def __init__(self, grid: GridWorld):
        self.grid = grid
        self.reservation_table = ReservationTable(reservations={}, robot_cell_dict={})
        self.planner = SIPPwRT(grid=grid, 
                               reservation_table=self.reservation_table)

    def obtain_path_for_agent(self,
                        start_pos: Coordinate,
                        goal_pos: Coordinate,
                        robot_profile: RobotProfile,
                        current_time: float = 0.0,
                        horizon: float = float('inf')) -> Optional[List[Tuple[Coordinate, TimeInterval]]]:
        path = self.planner.plan_path(start_pos=start_pos,
                                        goal_pos=goal_pos,
                                        robot_profile=robot_profile,
                                        current_time=current_time,
                                        horizon=horizon)

        return path

    def _reserve_cells_for_time_interval(self,
                                          start: Coordinate,
                                          end: Coordinate,
                                          time_interval: TimeInterval,
                                          robot_profile: RobotProfile) -> None:
        robot_reservations = self.grid.get_robot_reservations_for_move(robot_start_pos=start,
                                                                        robot_end_pos=end,
                                                                        robot_profile=robot_profile,
                                                                        current_time=time_interval.start,
                                                                        end_time=time_interval.end)
        for robot_reservation in robot_reservations:
            for cell in robot_reservation.robot_occupancy.occupied_cells:
                time_reservation = TimeReservation(interval=robot_reservation.time_interval,
                                                    robot_id=robot_profile.robot_id)
                self.reservation_table.add_reservation(cell, time_reservation)

    def _reserve_path(self,
                      path: List[Tuple[Coordinate, TimeInterval]],
                      robot_profile: RobotProfile) -> None:

        # Add reservations to the reservation table
        for i in range(len(path) - 1):
            start, start_time_interval = path[i]
            end, end_time_interval = path[i + 1]
            if start_time_interval.end > start_time_interval.start:
                self._reserve_cells_for_time_interval(start=start,
                                                      end=start,
                                                      time_interval=TimeInterval(start=start_time_interval.start,
                                                                                  end=start_time_interval.end),
                                                      robot_profile=robot_profile)
            
            self._reserve_cells_for_time_interval(start=start,
                                                  end=end,
                                                  time_interval=TimeInterval(start=start_time_interval.end,
                                                                              end=end_time_interval.start),
                                                  robot_profile=robot_profile)
        # Reserve the last position indefinitely
        last_pos, last_time_interval = path[-1]
        self._reserve_cells_for_time_interval(start=last_pos,
                                              end=last_pos,
                                              time_interval=TimeInterval(start=last_time_interval.start,
                                                                          end=float('inf')),
                                              robot_profile=robot_profile)

    def reserve_path_for_agent(self,
                               path: List[Tuple[Coordinate, TimeInterval]],
                               robot_profile: RobotProfile) -> None:
        self._reserve_path(path=path,
                           robot_profile=robot_profile)
    
    def plot_paths(self, 
                   paths: List[List[Tuple[Coordinate, TimeInterval]]], 
                   robot_radius: float) -> None:
        # Plot the grid and the paths without using grid object's plot method
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.grid.width * self.grid.cell_size)
        ax.set_ylim(0, self.grid.height * self.grid.cell_size)
        ax.set_aspect('equal')

        # Plot each path
        colors = plt.cm.get_cmap('hsv', len(paths) + 1)
        for i, path in enumerate(paths):
            for j in range(len(path) - 1):
                start, start_time = path[j]
                end, end_time = path[j + 1]
                ax.plot([start.x, end.x], [start.y, end.y], color=colors(i), linewidth=2)
                circle = Circle((start.x, start.y), robot_radius, color=colors(i), alpha=0.3)
                ax.add_patch(circle)
            # Draw the last position
            end, end_time = path[-1]
            circle = Circle((end.x, end.y), robot_radius, color=colors(i), alpha=0.3)
            ax.add_patch(circle)

        plt.show()

if __name__ == "__main__":
    width, height = 50, 25
    cell_size = 2*0.03534
    world = GridWorld.empty(width, height, cell_size)

    robot_profile = RobotProfile(radius=0.20, velocity=0.1, robot_id=1)
    start_pos = Coordinate(30*cell_size, 10*cell_size)

    goal_pos = Coordinate(5*cell_size, 20*cell_size)
    planner = MotionPlanner(grid=world)
    path = planner.obtain_path_for_agent(start_pos=start_pos,
                                        goal_pos=goal_pos,
                                        robot_profile=robot_profile,
                                        current_time=0.0,
                                        horizon=50.0)
    if path:
        print("Planned Path:")
        for pos, time_interval in path:
            print(f"Position: ({pos.x:.2f}, {pos.y:.2f}), Time: [{time_interval.start:.2f}, {time_interval.end:.2f}]")
        planner.reserve_path_for_agent(path=path, robot_profile=robot_profile)
        planner.plot_paths(paths=[path], robot_radius=robot_profile.radius)