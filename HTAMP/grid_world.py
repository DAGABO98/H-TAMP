import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Any
from matplotlib.patches import Rectangle, Circle

@dataclass(frozen=True, slots=True)
class Coordinate:
    x: float
    y: float
    tol: float = 1e-3

    def __post_init__(self):
        if self.tol <= 0:
            raise ValueError("tol must be > 0")

    def _key(self):
        # snap-to-grid; points in the same cell are "equal"
        return (round(self.x / self.tol), round(self.y / self.tol), self.tol)

    def __eq__(self, other):
        if not isinstance(other, Coordinate):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

@dataclass(frozen=True)
class GridIndex:
    index_x: int
    index_y: int

@dataclass(frozen=True)
class Cell: 
    lower_x: float
    lower_y: float
    upper_x: float
    upper_y: float

@dataclass(frozen=True)
class PosChange:
    dev_x: float
    dev_y: float

@dataclass(frozen=True)
class BoundingIndices:
    lower_x: int
    lower_y: int
    upper_x: int
    upper_y: int

@dataclass(frozen=True, slots=True)
class TimeInterval:
    start: float
    end: float
    tol: float = 1e-3

    def __post_init__(self):
        if self.tol <= 0:
            raise ValueError("tol must be > 0")

    def _key(self):
        # snap-to-grid; points in the same cell are "equal"
        return (round(self.start / self.tol), round(self.end / self.tol), self.tol)

    def __eq__(self, other):
        if not isinstance(other, TimeInterval):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

@dataclass
class RobotOccupancy:
    occupied_cells: Set[GridIndex]
    robot_center: Coordinate

@dataclass
class MotionReservation:
    time_interval: TimeInterval
    robot_occupancy: RobotOccupancy

@dataclass
class RobotProfile:
    radius: float
    speed: float
    robot_id: int

@dataclass
class GridWorld:
    width: int
    height: int
    cell_size: float  # meters per cell
    occupancy_map: np.ndarray  # 2D numpy array with 1=occupied, 0=free

    @classmethod
    def empty(cls, width: int, height: int, cell_size: float) -> "GridWorld":
        occupancy_map = np.zeros((height, width), dtype=np.uint8)
        return cls(width, height, cell_size, occupancy_map)

    def is_in_bounds_cell(self, 
                          cell_index: GridIndex) -> bool:
        return 0 <= cell_index.index_x < self.width and 0 <= cell_index.index_y < self.height

    def cell_rect(self, 
                  cell_index: GridIndex) -> Cell:
        lower_x = cell_index.index_x * self.cell_size
        lower_y = cell_index.index_y * self.cell_size
        return Cell(lower_x, lower_y, lower_x + self.cell_size, lower_y + self.cell_size)
    
    def get_cell_index(self, 
                       position: Coordinate) -> GridIndex:
        index_x = int(np.floor(position.x / self.cell_size))
        index_y = int(np.floor(position.y / self.cell_size))
        return GridIndex(index_x, index_y)
    
    def _generate_closest_cell_coordinate(self, 
                                        robot_center: Coordinate,
                                        cell_index: GridIndex) -> Coordinate:
        cell = self.cell_rect(cell_index)
        selected_x = min(max(robot_center.x, cell.lower_x), cell.upper_x)
        selected_y = min(max(robot_center.y, cell.lower_y), cell.upper_y)

        return Coordinate(selected_x, selected_y)

    def robot_intersects_cell(self, 
                              robot_center: Coordinate, 
                              robot_profile: RobotProfile,
                              cell_index: GridIndex) -> bool:
        selected_coordinate = self._generate_closest_cell_coordinate(robot_center,
                                                                    cell_index)
        dev_x = robot_center.x - selected_coordinate.x
        dev_y = robot_center.y - selected_coordinate.y
        return ((dev_x * dev_x) + (dev_y * dev_y)) <= (robot_profile.radius * robot_profile.radius)
    
    def _get_robot_bounding_indices(self, 
                                    robot_center: Coordinate, 
                                    robot_profile: RobotProfile) -> BoundingIndices:
        lower_index = self.get_cell_index(Coordinate(robot_center.x - robot_profile.radius, 
                                                     robot_center.y - robot_profile.radius))
        upper_index = self.get_cell_index(Coordinate(robot_center.x + robot_profile.radius, 
                                                     robot_center.y + robot_profile.radius))

        return BoundingIndices(lower_index.index_x, lower_index.index_y, 
                               upper_index.index_x, upper_index.index_y)

    def is_robot_in_bounds(self, 
                          robot_center: Coordinate, 
                          robot_profile: RobotProfile) -> bool:
        bounding_indices = self._get_robot_bounding_indices(robot_center, robot_profile)

        return (0 <= bounding_indices.lower_x < self.width and
                0 <= bounding_indices.lower_y < self.height and
                0 <= bounding_indices.upper_x < self.width and
                0 <= bounding_indices.upper_y < self.height)

    def get_occupied_cells_for_robot(self, 
                                     robot_center: Coordinate, 
                                     robot_profile: RobotProfile) -> List[GridIndex]:
        bounding_indices = self._get_robot_bounding_indices(robot_center, robot_profile)

        occupied_cells = []
        for index_x in range(bounding_indices.lower_x, bounding_indices.upper_x + 1):
            for index_y in range(bounding_indices.lower_y, bounding_indices.upper_y + 1):
                cell_index = GridIndex(index_x, index_y)
                if self.is_in_bounds_cell(cell_index):
                    if self.robot_intersects_cell(robot_center, robot_profile, cell_index):
                        occupied_cells.append(cell_index)
        return occupied_cells
    
    def is_robot_in_free_space(self, 
                                robot_center: Coordinate, 
                                robot_profile: RobotProfile) -> bool:
        if not self.is_robot_in_bounds(robot_center, robot_profile):
            return False

        occupied_cells = self.get_occupied_cells_for_robot(robot_center, robot_profile)
        for cell_index in occupied_cells:
            if self.occupancy_map[cell_index.index_y, cell_index.index_x] == 1:
                return False
        return True
    
    def _get_occupied_cells_for_partial_move(self, 
                                          robot_start_pos: Coordinate, 
                                          robot_profile: RobotProfile,
                                          pos_change: PosChange) -> Set[GridIndex]:
        robot_end_pos = Coordinate(robot_start_pos.x + pos_change.dev_x, 
                                   robot_start_pos.y + pos_change.dev_y)
        return set(self.get_occupied_cells_for_robot(robot_end_pos, robot_profile))
    
    def _get_occupied_cells_at_step(self,
                                    start_pos: Coordinate,
                                    robot_profile: RobotProfile,
                                    sign_num_minor_steps: int,
                                    sign_num_major_steps: int,
                                    scale_ratio: float,
                                    step_num: int,
                                    major_y: bool) -> Tuple[Set[GridIndex], PosChange]:
        minor_displacement = sign_num_minor_steps * scale_ratio * step_num * self.cell_size
        major_displacement = sign_num_major_steps * step_num * self.cell_size
        if major_y:
            pos_change = PosChange(dev_x=minor_displacement, dev_y=major_displacement)
        else:
            pos_change = PosChange(dev_x=major_displacement, dev_y=minor_displacement)

        occupied_cells: Set[GridIndex] = self._get_occupied_cells_for_partial_move(robot_start_pos=start_pos, 
                                                                                    robot_profile=robot_profile, 
                                                                                    pos_change=pos_change)
        return occupied_cells, pos_change
    
    def _generate_robot_occupancy_list(self,
                                       start_pos: Coordinate, 
                                       robot_profile: RobotProfile,
                                       abs_num_major_steps: int,
                                       abs_num_minor_steps: int,
                                       sign_num_major_steps: int,
                                       sign_num_minor_steps: int,
                                       major_y: bool) -> List[RobotOccupancy]:
        robot_occupancies: List[RobotOccupancy] = []

        scale_ratio = float(abs_num_minor_steps) / float(abs_num_major_steps)

        for i in range(abs_num_major_steps):
            prev_cells, _ = self._get_occupied_cells_at_step(start_pos=start_pos,
                                                            robot_profile=robot_profile,
                                                            sign_num_minor_steps=sign_num_minor_steps,
                                                            sign_num_major_steps=sign_num_major_steps,
                                                            scale_ratio=scale_ratio,
                                                            step_num=i,
                                                            major_y=major_y)
            
            curr_cells, curr_pos_change = self._get_occupied_cells_at_step(start_pos=start_pos,
                                                        robot_profile=robot_profile,
                                                        sign_num_minor_steps=sign_num_minor_steps,
                                                        sign_num_major_steps=sign_num_major_steps,
                                                        scale_ratio=scale_ratio,
                                                        step_num=i+1,
                                                        major_y=major_y)

            occupied_cells = prev_cells.union(curr_cells)

            robot_occupancy = RobotOccupancy(occupied_cells=occupied_cells,
                                             robot_center=Coordinate(start_pos.x + curr_pos_change.dev_x, 
                                                                     start_pos.y + curr_pos_change.dev_y))
            robot_occupancies.append(robot_occupancy)
        return robot_occupancies
    
    def _generate_robot_timing_list(self,
                                   robot_profile: RobotProfile,
                                   current_time: float,
                                   abs_num_major_steps: int,
                                   abs_num_minor_steps: int,
                                   sign_num_major_steps: int,
                                   sign_num_minor_steps: int) -> List[TimeInterval]:
        time_intervals: List[TimeInterval] = []

        scale_ratio = float(abs_num_minor_steps) / float(abs_num_major_steps)

        for i in range(abs_num_major_steps):
            prev_minor_displacement = sign_num_minor_steps * scale_ratio * i * self.cell_size
            prev_major_displacement = sign_num_major_steps * i * self.cell_size
            prev_total_displacement = np.sqrt(prev_minor_displacement**2 + prev_major_displacement**2)
            prev_time_to_end = prev_total_displacement / robot_profile.speed if robot_profile.speed > 0 else 0

            minor_displacement = sign_num_minor_steps * scale_ratio * (i+1) * self.cell_size
            major_displacement = sign_num_major_steps * (i+1) * self.cell_size
            total_displacement = np.sqrt(minor_displacement**2 + major_displacement**2)
            time_to_end = total_displacement / robot_profile.speed if robot_profile.speed > 0 else 0
            
            time_interval = TimeInterval(start=current_time + prev_time_to_end, 
                                         end=current_time + time_to_end)
            time_intervals.append(time_interval)
        return time_intervals
    
    def get_robot_occupancy_for_move(self,
                                     robot_start_pos: Coordinate, 
                                     robot_end_pos: Coordinate,
                                     robot_profile: RobotProfile) -> List[RobotOccupancy]:
        num_y_steps = round((robot_end_pos.y - robot_start_pos.y) / self.cell_size)
        num_x_steps = round((robot_end_pos.x - robot_start_pos.x) / self.cell_size)

        if num_y_steps == 0 and num_x_steps == 0:
            occupied_cells = set(self.get_occupied_cells_for_robot(robot_center=robot_start_pos,
                                                                   robot_profile=robot_profile))
            return [RobotOccupancy(occupied_cells=occupied_cells,
                                   robot_center=robot_start_pos)]

        abs_num_x_steps = abs(num_x_steps)
        abs_num_y_steps = abs(num_y_steps)
        sign_num_x_steps = 1 if num_x_steps > 0 else -1
        sign_num_y_steps = 1 if num_y_steps > 0 else -1

        if abs_num_y_steps > abs_num_x_steps:
            major_y = True
            return self._generate_robot_occupancy_list(start_pos=robot_start_pos, 
                                                       robot_profile=robot_profile,
                                                       abs_num_major_steps=abs_num_y_steps,
                                                       abs_num_minor_steps=abs_num_x_steps,
                                                       sign_num_major_steps=sign_num_y_steps,
                                                       sign_num_minor_steps=sign_num_x_steps,
                                                       major_y=major_y)
        else:
            major_y = False
            return self._generate_robot_occupancy_list(start_pos=robot_start_pos, 
                                                       robot_profile=robot_profile,
                                                       abs_num_major_steps=abs_num_x_steps,
                                                       abs_num_minor_steps=abs_num_y_steps,
                                                       sign_num_major_steps=sign_num_x_steps,
                                                       sign_num_minor_steps=sign_num_y_steps,
                                                       major_y=major_y)
    
    def get_robot_timing_for_move(self,
                                  robot_start_pos: Coordinate, 
                                  robot_end_pos: Coordinate,
                                  robot_profile: RobotProfile,
                                  current_time: float,
                                  end_time: float) -> List[TimeInterval]:
        num_y_steps = round((robot_end_pos.y - robot_start_pos.y) / self.cell_size)
        num_x_steps = round((robot_end_pos.x - robot_start_pos.x) / self.cell_size)

        if num_y_steps == 0 and num_x_steps == 0:
            return [TimeInterval(start=current_time, end=end_time)]

        abs_num_x_steps = abs(num_x_steps)
        abs_num_y_steps = abs(num_y_steps)
        sign_num_x_steps = 1 if num_x_steps > 0 else -1
        sign_num_y_steps = 1 if num_y_steps > 0 else -1

        if abs_num_y_steps > abs_num_x_steps:
            return self._generate_robot_timing_list(robot_profile=robot_profile,
                                                    current_time=current_time,
                                                    abs_num_major_steps=abs_num_y_steps,
                                                    abs_num_minor_steps=abs_num_x_steps,
                                                    sign_num_major_steps=sign_num_y_steps,
                                                    sign_num_minor_steps=sign_num_x_steps)
        else:
            return self._generate_robot_timing_list(robot_profile=robot_profile,
                                                    current_time=current_time,
                                                    abs_num_major_steps=abs_num_x_steps,
                                                    abs_num_minor_steps=abs_num_y_steps,
                                                    sign_num_major_steps=sign_num_x_steps,
                                                    sign_num_minor_steps=sign_num_y_steps)

    def get_robot_reservations_for_move(self, 
                                  robot_start_pos: Coordinate, 
                                  robot_end_pos: Coordinate,
                                  robot_profile: RobotProfile,
                                  current_time: float, 
                                  end_time: float) -> List[MotionReservation]:
        
        robot_occupancies = self.get_robot_occupancy_for_move(robot_start_pos, 
                                                             robot_end_pos,
                                                             robot_profile)
        time_intervals = self.get_robot_timing_for_move(robot_start_pos,
                                                        robot_end_pos,
                                                        robot_profile,
                                                        current_time,
                                                        end_time)
        assert len(robot_occupancies) == len(time_intervals)
        reservations: List[MotionReservation] = []
        for i in range(len(robot_occupancies)):
            reservation = MotionReservation(time_interval=time_intervals[i],
                                            robot_occupancy=robot_occupancies[i])
            reservations.append(reservation)

        return reservations
    
    def is_move_collision_free(self,
                               robot_start_pos: Coordinate, 
                               robot_end_pos: Coordinate,
                               robot_profile: RobotProfile) -> bool:
        robot_occupancies = self.get_robot_occupancy_for_move(robot_start_pos, 
                                                             robot_end_pos,
                                                             robot_profile)
        for occupancy in robot_occupancies:
            for cell_index in occupancy.occupied_cells:
                if self.occupancy_map[cell_index.index_y, cell_index.index_x] == 1:
                    return False
        return True
    
    def _get_potential_move_position(self,
                                    robot_center: Coordinate, 
                                    robot_profile: RobotProfile,
                                    index_x: int, 
                                    index_y: int) -> Coordinate:
        cell_index = GridIndex(index_x, index_y)
        coordinate = self._generate_closest_cell_coordinate(robot_center=robot_center,
                                                            cell_index=cell_index)

        if not self.is_robot_in_bounds(coordinate, robot_profile) or \
            not self.is_robot_in_free_space(coordinate, robot_profile) or \
                not self.is_move_collision_free(robot_center, coordinate, robot_profile):
            return None
        else:
            return coordinate

    def get_valid_moves(self, 
                        robot_center: Coordinate, 
                        robot_profile: RobotProfile) -> List[Coordinate]:
        valid_moves: List[Coordinate] = []

        single_x_displacement = [1.0, 0.0, -1.0]
        single_y_displacement = [1.0, 0.0, -1.0]
        double_x_displacement = [2.0, -2.0]
        double_y_displacement = [2.0, -2.0]

        for single_x in single_x_displacement:
            for single_y in single_y_displacement:
                if single_x == 0 and single_y == 0:
                    continue
                coordinate = Coordinate(robot_center.x + (single_x * self.cell_size),
                                        robot_center.y + (single_y * self.cell_size))
                if self.is_robot_in_bounds(coordinate, robot_profile) and \
                    self.is_robot_in_free_space(coordinate, robot_profile) and \
                        self.is_move_collision_free(robot_center, coordinate, robot_profile):
                    valid_moves.append(coordinate)

        for double_x in double_x_displacement:
            for single_y in single_y_displacement:
                if single_y == 0:
                    continue
                coordinate = Coordinate(robot_center.x + (double_x * self.cell_size),
                                        robot_center.y + (single_y * self.cell_size))
                if self.is_robot_in_bounds(coordinate, robot_profile) and \
                    self.is_robot_in_free_space(coordinate, robot_profile) and \
                        self.is_move_collision_free(robot_center, coordinate, robot_profile):   
                    valid_moves.append(coordinate)
        
        for double_y in double_y_displacement:
            for single_x in single_x_displacement:
                if single_x == 0:
                    continue
                coordinate = Coordinate(robot_center.x + single_x * self.cell_size,
                                        robot_center.y + double_y * self.cell_size)
                if self.is_robot_in_bounds(coordinate, robot_profile) and \
                    self.is_robot_in_free_space(coordinate, robot_profile) and \
                        self.is_move_collision_free(robot_center, coordinate, robot_profile):
                    valid_moves.append(coordinate)

        return valid_moves
    
    def plot_next_move(self, 
                       cell_size: float = 0.03534*2, 
                       robot_profile: RobotProfile = RobotProfile(radius=0.20, speed=0.1, robot_id=1),
                       start_pos: Coordinate = Coordinate(30*0.03534*2, 10*0.03534*2),
                       end_pos: Coordinate = Coordinate(32*0.03534*2, 10*0.03534*2), 
                       next_positions: List[Coordinate] = []):
        fig, ax = plt.subplots(figsize=(8, 6))
        for x in np.arange(0, (self.width + 1) * cell_size, cell_size):
            ax.axvline(x, linewidth=0.3)
        for y in np.arange(0, (self.height + 1) * cell_size, cell_size):
            ax.axhline(y, linewidth=0.3)

        for grid_index in self.get_occupied_cells_for_robot(robot_center=start_pos, 
                                                            robot_profile=robot_profile):
            cell = self.cell_rect(cell_index=grid_index)
            ax.add_patch(Rectangle((cell.lower_x, cell.lower_y), cell_size, cell_size, fill=False, linewidth=1.2))

        # Plot original robot position
        ax.add_patch(Circle((start_pos.x, start_pos.y), robot_profile.radius, fill=False, color='blue', linewidth=2, label='Original Position'))

        # Plot new robot position
        ax.add_patch(Circle((end_pos.x, end_pos.y), robot_profile.radius, fill=False, color='green', linewidth=2, label='New Position'))

        # Plot the line of movement
        ax.plot([start_pos.x, end_pos.x], [start_pos.y, end_pos.y], 'r--', linewidth=1, label='Movement')


        preview_radius = robot_profile.radius * 0.05
        for coordinate in next_positions:
            ax.add_patch(Circle((coordinate.x, coordinate.y), preview_radius, color='red', fill=True))

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, width * cell_size)
        ax.set_ylim(0, height * cell_size)
        ax.set_title("Grid World • Circular Robot Covering Multiple Cells • Potential Moves")
        ax.legend()
        plt.savefig("results/motion_planning/grid_move_example.png")
        plt.close()

    def plot_reservations(self, 
                          reservations: List[MotionReservation], 
                          cell_size: float,
                          robot_profile: RobotProfile, 
                          start_pos: Coordinate, 
                          end_pos: Coordinate):

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot grid lines
        for x in np.arange(0, (self.width + 1) * cell_size, cell_size):
            ax.axvline(x, linewidth=0.3, color='lightgray')
        for y in np.arange(0, (self.height + 1) * cell_size, cell_size):
            ax.axhline(y, linewidth=0.3, color='lightgray')
        
        # Plot original robot position
        ax.add_patch(Circle((start_pos.x, start_pos.y), robot_profile.radius, fill=False, color='blue', linewidth=1, label='Original Position'))

        # Plot new robot position
        ax.add_patch(Circle((end_pos.x, end_pos.y), robot_profile.radius, fill=False, color='green', linewidth=1, label='New Position'))

        # Plot the line of movement
        ax.plot([start_pos.x, end_pos.x], [start_pos.y, end_pos.y], 'r--', linewidth=1, label='Movement')

        # Plot reserved cells with different colors for different time intervals
        color_map = plt.get_cmap('viridis', len(reservations))
        for i, reservation in enumerate(reservations):
            print(i)
            color = color_map(i)
            for cell_index in reservation.robot_occupancy.occupied_cells:
                cell = self.cell_rect(cell_index=cell_index)
                ax.add_patch(Rectangle((cell.lower_x, cell.lower_y), cell_size, cell_size, color=color, alpha=0.2, 
                                       label=f'{reservation.time_interval.start:.2f}-{reservation.time_interval.end:.2f}'))

            ax.add_patch(Circle((reservation.robot_occupancy.robot_center.x, 
                                 reservation.robot_occupancy.robot_center.y), 
                                 robot_profile.radius, fill=False, color=color, linewidth=0.5))

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, self.width * cell_size)
        ax.set_ylim(0, self.height * cell_size)
        ax.set_title("Grid World • Cell Reservations Over Time")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")

        # Create a legend with unique labels
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(dict.fromkeys(labels))
        unique_handles = [handles[labels.index(ul)] for ul in unique_labels]
        ax.legend(unique_handles, unique_labels, title="Time Intervals (s)")

        plt.savefig("results/motion_planning/grid_reservations_example.png")
        plt.close()

if __name__ == "__main__":
    width, height = 50, 25
    cell_size = 2*0.03534
    world = GridWorld.empty(width, height, cell_size)

    robot_profile = RobotProfile(radius=0.20, speed=0.1, robot_id=1)
    start_pos = Coordinate(30*cell_size, 10*cell_size)

    occ_cells = world.get_occupied_cells_for_robot(robot_center=start_pos, 
                                                   robot_profile=robot_profile)
    next_positions = world.get_valid_moves(robot_center=start_pos, 
                                           robot_profile=robot_profile)
    #Select a random valid move
    import random
    if next_positions:
        end_pos = random.choice(next_positions)
    else:
        end_pos = start_pos # Stay in place if no valid moves
    
    world.plot_next_move(cell_size=cell_size, 
                         robot_profile=robot_profile, 
                         start_pos=start_pos, 
                         end_pos=end_pos,
                         next_positions=next_positions)
    reservations = world.get_robot_reservations_for_move(robot_start_pos=start_pos, 
                                                         robot_end_pos=end_pos, 
                                                         robot_profile=robot_profile, 
                                                         current_time=0.0,
                                                         end_time=5.0)
    world.plot_reservations(reservations, cell_size, robot_profile, start_pos, end_pos)

    
        
