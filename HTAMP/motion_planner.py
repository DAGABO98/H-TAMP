import argparse
from datetime import datetime
import heapq
import random
import traceback
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict
from matplotlib.patches import Circle

from HTAMP.loc_dataclasses import Coordinate
from HTAMP.grid_world import TimeInterval, GridWorld, GridIndex, RobotProfile
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter
from HTAMP.traversal_dataclasses import TraversalNode
from HTAMP.traversal_graph_gen import TraversalGraphGenerator

@dataclass
class TimeReservation:
    interval: TimeInterval
    robot_id: int

    def overlaps(self, other: "TimeReservation") -> bool:
        return round(self.interval.start, 4) <= round(other.interval.end, 4) and \
            round(self.interval.end, 4) >= round(other.interval.start, 4)
    
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
    f: float
    g: float
    node: "SIPPNode" = field(compare=False)

@dataclass
class SIPPNode:
    traversal_node: TraversalNode
    interval: TimeInterval
    arrival: float
    parent: Optional["SIPPNode"] = None

class SIPPwRT:
    def __init__(self, 
                 grid: GridWorld, 
                 reservation_table: Optional[ReservationTable],
                 weight_factor: float = 1.0):
        self.grid = grid
        self.reservation_table = reservation_table
        self.weight_factor = weight_factor

    def heuristic(self, 
                  start_traversal_node: TraversalNode, 
                  goal_traversal_node: TraversalNode,
                  robot_profile: RobotProfile) -> float:
        _, distance = self.grid.get_shortest_path(start=start_traversal_node, goal=goal_traversal_node)

        time_to_goal = distance / robot_profile.speed if robot_profile.speed > 0 else float('inf')
        return time_to_goal

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
    
    def _get_safe_intervals_for_current_node(self,
                                             current_traversal_node: TraversalNode,
                                             robot_profile: RobotProfile,
                                             horizon: float = float('inf')) -> List[TimeReservation]:
        safe_intervals: List[TimeReservation] = []
        occupied_cells: Set[GridIndex] = self.grid._get_occupied_cells_for_static_position(robot_position=current_traversal_node.position,
                                                                                           robot_profile=robot_profile)

        for cell in occupied_cells:
            cell_safe_intervals = self.reservation_table.get_safe_intervals(cell=cell, 
                                                                            horizon=horizon, 
                                                                            robot_id=robot_profile.robot_id)
            if not safe_intervals:
                safe_intervals = cell_safe_intervals
            else:
                safe_intervals = self._intersect_intervals(safe_intervals, 
                                                           cell_safe_intervals)

        return safe_intervals

    def _get_safe_intervals_for_move(self, 
                                     from_traversal_node: TraversalNode,
                                     to_traversal_node: TraversalNode,
                                     robot_profile: RobotProfile,
                                     horizon: float = float('inf')) -> List[TimeReservation]:
        safe_intervals: List[TimeReservation] = []
        robot_occupancy_list = self.grid.get_robot_occupancy_for_move(from_node=from_traversal_node,
                                                                      to_node=to_traversal_node,
                                                                      robot_profile=robot_profile)
        occupied_cells: Set[GridIndex] = set()
        for robot_occupancy in robot_occupancy_list:
            occupied_cells.update(robot_occupancy.occupied_cells)

        for cell in occupied_cells:
            cell_safe_intervals = self.reservation_table.get_safe_intervals(cell=cell, 
                                                                            horizon=horizon, 
                                                                            robot_id=robot_profile.robot_id)
            if not safe_intervals:
                safe_intervals = cell_safe_intervals
            else:
                safe_intervals = self._intersect_intervals(safe_intervals, 
                                                           cell_safe_intervals)

        return safe_intervals
    
    def check_conflict_for_move(self,
                                from_traversal_node: TraversalNode,
                                to_traversal_node: TraversalNode,
                                robot_profile: RobotProfile, 
                                current_time: float) -> bool:

        robot_reservations = self.grid.get_robot_reservations_for_move(from_node=from_traversal_node,
                                                                       to_node=to_traversal_node,
                                                                       robot_profile=robot_profile,
                                                                       current_time=current_time)
        
        for robot_reservation in robot_reservations:
            for cell in robot_reservation.robot_occupancy.occupied_cells:
                time_reservation = TimeReservation(interval=robot_reservation.time_interval, 
                                                   robot_id=robot_profile.robot_id)
                if self.reservation_table.check_conflict(cell, time_reservation):
                    return True
        return False

    def _get_travel_time(self,
                         from_traversal_node: TraversalNode,
                         to_traversal_node: TraversalNode,
                         robot_profile: RobotProfile) -> float:
        current_edge = self.grid.traversal_graph.edge_dict.get((from_traversal_node.label, to_traversal_node.label))
        if current_edge is None:
            raise ValueError(f"No edge found between {from_traversal_node.label} and {to_traversal_node.label}")
        distance = current_edge.edge_connector.length()
        travel_time = distance / robot_profile.speed if robot_profile.speed > 0 else float('inf')
        return travel_time

    def _get_earliest_departure(self,
                                curr_node: SIPPNode,
                                from_traversal_node: TraversalNode,
                                to_traversal_node: TraversalNode,
                                robot_profile: RobotProfile,
                                end_pos_interval: TimeInterval) -> Optional[float]:
        # Find the earliest departure time for the current node
        current_time = max(curr_node.arrival, curr_node.interval.start)
        travel_time = self._get_travel_time(from_traversal_node=from_traversal_node, 
                                             to_traversal_node=to_traversal_node, 
                                             robot_profile=robot_profile)

        window_start = end_pos_interval.start - travel_time
        current_time = max(current_time, window_start)

        if self.check_conflict_for_move(from_traversal_node=from_traversal_node, 
                                        to_traversal_node=to_traversal_node,
                                        robot_profile=robot_profile,
                                        current_time=current_time):
            return None
        if current_time + travel_time <= end_pos_interval.end:
            return current_time
        return None

    def _reconstruct_path(self, 
                          sipp_node: SIPPNode, 
                          robot_profile: RobotProfile) -> List[Tuple[TraversalNode, TimeInterval]]:
        sipp_node_list : List[SIPPNode] = []
        while sipp_node:
            sipp_node_list.append(sipp_node)
            sipp_node = sipp_node.parent

        sipp_node_list.reverse()
        timed_path: List[Tuple[TraversalNode, TimeInterval]] = []
        for i in range(len(sipp_node_list) - 1):
            n = sipp_node_list[i]
            next_n = sipp_node_list[i + 1]
            travel_time = self._get_travel_time(n.traversal_node, next_n.traversal_node, robot_profile)
            departure_time = next_n.arrival - travel_time
            timed_path.append((n.traversal_node, TimeInterval(n.arrival, departure_time)))

        # Add the last node with an open-ended time interval
        last_sipp_node = sipp_node_list[-1]
        timed_path.append((last_sipp_node.traversal_node, TimeInterval(last_sipp_node.arrival, float('inf'))))

        return timed_path

    def plan_path(self, 
                  start_traversal_node: TraversalNode, 
                  goal_traversal_node: TraversalNode, 
                  robot_profile: RobotProfile,
                  current_time: float = 0.0,
                  horizon: float = float('inf')) -> Optional[List[Tuple[TraversalNode, TimeInterval]]]:
        # Plan the path from start to goal while avoiding obstacles
        start_safe_intervals = self._get_safe_intervals_for_current_node(start_traversal_node,
                                                                       robot_profile,
                                                                       horizon)

        assert start_safe_intervals, "No safe intervals at start position"

        sipp_node_list: List[SIPPNode] = []
        for time_reservation in start_safe_intervals:
            if time_reservation.interval.end < current_time:
                continue
            arrival_time = current_time
            sipp_node_list.append(SIPPNode(traversal_node=start_traversal_node, interval=time_reservation.interval, arrival=arrival_time))

        if not sipp_node_list:
            return None
        
        open_set: List[PQItem] = []
        seen_set: Dict[Tuple[str, TimeInterval], float] = {}
        for sipp_node in sipp_node_list:
            f = sipp_node.arrival + (self.weight_factor * self.heuristic(sipp_node.traversal_node, goal_traversal_node, robot_profile))
            heapq.heappush(open_set, PQItem(f=f, g=sipp_node.arrival, node=sipp_node))
            seen_set[(sipp_node.traversal_node.label, sipp_node.interval)] = sipp_node.arrival

        while open_set:
            current_item = heapq.heappop(open_set)
            current_sipp_node = current_item.node

            if current_sipp_node.traversal_node.label == goal_traversal_node.label:
                return self._reconstruct_path(sipp_node=current_sipp_node, 
                                              robot_profile=robot_profile)

            for next_traversal_node_label in current_sipp_node.traversal_node.connections:
                potential_next_move = self.grid.traversal_graph.nodes_dict[next_traversal_node_label]

                for safe_interval in self._get_safe_intervals_for_move(from_traversal_node=current_sipp_node.traversal_node,
                                                                      to_traversal_node=potential_next_move,
                                                                      robot_profile=robot_profile,
                                                                      horizon=horizon):

                    earliest_departure = self._get_earliest_departure(curr_node=current_sipp_node,
                                                                      from_traversal_node=current_sipp_node.traversal_node,
                                                                      to_traversal_node=potential_next_move,
                                                                      robot_profile=robot_profile,
                                                                      end_pos_interval=safe_interval.interval)
                    if earliest_departure is None: 
                        continue

                    travel_time = self._get_travel_time(current_sipp_node.traversal_node, potential_next_move, robot_profile)
                    arrival_time = earliest_departure + travel_time
                    child_node = SIPPNode(traversal_node=potential_next_move,
                                          interval=safe_interval.interval,
                                          arrival=arrival_time,
                                          parent=current_sipp_node)
                    child_key = (child_node.traversal_node.label, child_node.interval)
                    g_prev = seen_set.get(child_key)
                    if g_prev is None or arrival_time < g_prev:
                        seen_set[child_key] = arrival_time
                        f = arrival_time + (self.weight_factor * self.heuristic(child_node.traversal_node, goal_traversal_node, robot_profile))
                        heapq.heappush(open_set, PQItem(f=f, g=arrival_time, node=child_node))

        return None

class MotionPlanner:
    def __init__(self, grid: GridWorld, weight_factor: float = 1.0):
        self.grid = grid
        self.reservation_table = ReservationTable(reservations={}, robot_cell_dict={})
        self.planner = SIPPwRT(grid=grid, 
                               reservation_table=self.reservation_table,
                               weight_factor=weight_factor)

    def obtain_path_for_agent(self,
                              start_traversal_node: TraversalNode,
                              goal_traversal_node: TraversalNode,
                              robot_profile: RobotProfile,
                              current_time: float = 0.0,
                              horizon: float = float('inf')) -> Optional[List[Tuple[TraversalNode, TimeInterval]]]:
        path = self.planner.plan_path(start_traversal_node=start_traversal_node,
                                        goal_traversal_node=goal_traversal_node,
                                        robot_profile=robot_profile,
                                        current_time=current_time,
                                        horizon=horizon)

        return path

    def _reserve_cells_for_time_interval(self,
                                          from_node: TraversalNode,
                                          to_node: TraversalNode,
                                          time_interval: TimeInterval,
                                          robot_profile: RobotProfile) -> None:
        if from_node.label == to_node.label:
            occupied_cells = self.grid._get_occupied_cells_for_static_position(robot_position=from_node.position,
                                                                               robot_profile=robot_profile)
            for cell in occupied_cells:
                time_reservation = TimeReservation(interval=time_interval,
                                                   robot_id=robot_profile.robot_id)
                self.reservation_table.add_reservation(cell, time_reservation)
        else:
            robot_reservations = self.grid.get_robot_reservations_for_move(from_node=from_node,
                                                                            to_node=to_node,
                                                                            robot_profile=robot_profile,
                                                                            current_time=time_interval.start)
            for robot_reservation in robot_reservations:
                for cell in robot_reservation.robot_occupancy.occupied_cells:
                    time_reservation = TimeReservation(interval=robot_reservation.time_interval,
                                                        robot_id=robot_profile.robot_id)
                    self.reservation_table.add_reservation(cell, time_reservation)

    def _reserve_path(self,
                      path: List[Tuple[TraversalNode, TimeInterval]],
                      robot_profile: RobotProfile) -> None:

        # Add reservations to the reservation table
        for i in range(len(path) - 1):
            start_node, start_time_interval = path[i]
            end_node, end_time_interval = path[i + 1]
            if start_time_interval.end > start_time_interval.start:
                self._reserve_cells_for_time_interval(from_node=start_node,
                                                      to_node=start_node,
                                                      time_interval=TimeInterval(start=start_time_interval.start,
                                                                                  end=start_time_interval.end),
                                                      robot_profile=robot_profile)
            
            self._reserve_cells_for_time_interval(from_node=start_node,
                                                  to_node=end_node,
                                                  time_interval=TimeInterval(start=start_time_interval.end,
                                                                              end=end_time_interval.start),
                                                  robot_profile=robot_profile)
        # Reserve the last position indefinitely
        last_node, last_time_interval = path[-1]
        self._reserve_cells_for_time_interval(from_node=last_node,
                                              to_node=last_node,
                                              time_interval=TimeInterval(start=last_time_interval.start,
                                                                          end=float('inf')),
                                              robot_profile=robot_profile)

    def reserve_path_for_agent(self,
                               path: List[Tuple[Coordinate, TimeInterval]],
                               robot_profile: RobotProfile) -> None:
        self._reserve_path(path=path,
                           robot_profile=robot_profile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="maps/FA3/FA3_lanes.yaml", help="Path to the configuration file")
    parser.add_argument("--occupancy_map_path", type=str, default="maps/FA3/occupancy_map.npy", help="Path to the input occupancy map")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for the grid world")
    parser.add_argument("--occupancy_reservations_file", type=str, default="data/occupancy_reservations.pkl", help="Path to the occupancy reservations file")
    parser.add_argument("--use_saved_data", action='store_true', help="Whether to use saved occupancy reservations data")
    parser.add_argument("--num_robots", type=int, default=1, help="Number of robots to plan for")
    args = parser.parse_args()

    print("Generating Traversal Graph...")

    tg_generator = TraversalGraphGenerator(occupancy_map_path=args.occupancy_map_path,
                                           config_path=args.config_path,
                                           meters_per_pixel=args.meters_per_pixel,
                                           factor=args.factor)

    print("Traversal Graph generated.")

    potential_nodes = []

    for doorway in tg_generator.doorway_subgraphs:
        room_nodes = doorway.room_nodes
        for room_node_label in room_nodes:
            room_node = tg_generator.traversal_graph.nodes_dict[room_node_label]
            potential_nodes.append(room_node)
            

    robot_profiles = []

    # randomly select start and goal nodes for each robot
    selected_start_nodes = random.sample(potential_nodes, args.num_robots)
    selected_goal_nodes = random.sample(potential_nodes, args.num_robots)

    for i in range(args.num_robots):
        robot_profile = RobotProfile(radius=0.20, speed=0.20, robot_id=i+1)
        robot_profiles.append(robot_profile)

    print("Creating Grid World...")

    world = GridWorld(cell_size=tg_generator.meters_per_cell,
                      fps=args.fps,
                      occupancy_map=tg_generator.occupancy_map,
                      traversal_graph=tg_generator.traversal_graph,
                      shortest_paths=tg_generator.shortest_paths,
                      robot_profiles=robot_profiles,
                      use_saved_data=args.use_saved_data,
                      occupancy_reservations_file=args.occupancy_reservations_file)

    print("Grid World created.")

    planner = MotionPlanner(grid=world, weight_factor=1.0)

    paths = []

    pStart = datetime.now()

    for i in range(args.num_robots):
        path = planner.obtain_path_for_agent(start_traversal_node=selected_start_nodes[i],
                                            goal_traversal_node=selected_goal_nodes[i],
                                            robot_profile=robot_profiles[i],
                                            current_time=0.0,
                                            horizon=500.0)
        if path:
            print(f"Planned Path for Robot {i}:")
            for traversal_node, time_interval in path:
                print(f"Node: ({traversal_node.label}), Time: [{time_interval.start:.2f}, {time_interval.end:.2f}]")
            planner.reserve_path_for_agent(path=path, robot_profile=robot_profile)
            paths.append(path)

    pEnd = datetime.now()
    print(f"Total planning time: {pEnd - pStart}")

    MotionPlanningPlotter.plot_paths(occupancy_map=tg_generator.occupancy_map,
                                    origin_x=tg_generator.origin_x,
                                    origin_y=tg_generator.origin_y,
                                    resolution=tg_generator.meters_per_cell,
                                    paths=paths,
                                    traversal_graph=tg_generator.traversal_graph,
                                    robot_profiles=robot_profiles)

if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")