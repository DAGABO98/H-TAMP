import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Any
from matplotlib.patches import Rectangle, Circle

@dataclass
class Coordinate:
    x: float
    y: float

@dataclass
class GridIndex:
    index_x: int
    index_y: int

@dataclass
class Cell:
    lower_x: float
    lower_y: float
    upper_x: float
    upper_y: float

@dataclass
class PosChange:
    dev_x: float
    dev_y: float

@dataclass
class BoundingIndices:
    lower_x: int
    lower_y: int
    upper_x: int
    upper_y: int

@dataclass
class TimeInterval:
    start: float
    end: float

@dataclass
class RobotOccupancy:
    occupied_cells: Set[GridIndex]
    robot_center: Coordinate
    robot_radius: float

@dataclass
class MotionReservation:
    time_interval: TimeInterval
    robot_occupancy: RobotOccupancy

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

    def robot_intersects_cell(self, 
                              robot_center: Coordinate, 
                              robot_radius: float, 
                              cell_index: GridIndex) -> bool:
        cell = self.cell_rect(cell_index)
        selected_x = min(max(robot_center.x, cell.lower_x), cell.upper_x)
        selected_y = min(max(robot_center.y, cell.lower_y), cell.upper_y)
        dev_x = robot_center.x - selected_x
        dev_y = robot_center.y - selected_y
        return ((dev_x * dev_x) + (dev_y * dev_y)) <= (robot_radius * robot_radius)
    
    def _get_robot_bounding_indices(self, 
                                    robot_center: Coordinate, 
                                    robot_radius: float) -> BoundingIndices:
        lower_index_x = int(np.floor((robot_center.x - robot_radius) / self.cell_size))
        lower_index_y = int(np.floor((robot_center.y - robot_radius) / self.cell_size))
        upper_index_x = int(np.floor((robot_center.x + robot_radius) / self.cell_size))
        upper_index_y = int(np.floor((robot_center.y + robot_radius) / self.cell_size))

        return BoundingIndices(lower_index_x, lower_index_y, upper_index_x, upper_index_y)

    def is_robot_in_bounds(self, 
                          robot_center: Coordinate, 
                          robot_radius: float) -> bool:
        bounding_indices = self._get_robot_bounding_indices(robot_center, robot_radius)

        return (0 <= bounding_indices.lower_x < self.width and
                0 <= bounding_indices.lower_y < self.height and
                0 <= bounding_indices.upper_x < self.width and
                0 <= bounding_indices.upper_y < self.height)

    def get_occupied_cells_for_robot(self, 
                                     robot_center: Coordinate, 
                                     robot_radius: float) -> List[GridIndex]:
        bounding_indices = self._get_robot_bounding_indices(robot_center, robot_radius)

        occupied_cells = []
        for index_x in range(bounding_indices.lower_x, bounding_indices.upper_x + 1):
            for index_y in range(bounding_indices.lower_y, bounding_indices.upper_y + 1):
                cell_index = GridIndex(index_x, index_y)
                if self.is_in_bounds_cell(cell_index):
                    if self.robot_intersects_cell(robot_center, robot_radius, cell_index):
                        occupied_cells.append(cell_index)
        return occupied_cells
    
    def is_robot_collision_free(self, 
                                robot_center: Coordinate, 
                                robot_radius: float) -> bool:
        if not self.is_robot_in_bounds(robot_center, robot_radius):
            return False

        occupied_cells = self.get_occupied_cells_for_robot(robot_center, robot_radius)
        for cell_index in occupied_cells:
            if self.occupancy_map[cell_index.index_y, cell_index.index_x] == 1:
                return False
        return True
    
    def _generate_potential_move(self, 
                                robot_center: Coordinate, 
                                robot_radius: float, 
                                cell_index: GridIndex) -> Coordinate:
        cell = self.cell_rect(cell_index)
        selected_x = min(max(robot_center.x, cell.lower_x), cell.upper_x)
        selected_y = min(max(robot_center.y, cell.lower_y), cell.upper_y)
        coordinate = Coordinate(selected_x, selected_y)

        if self.is_robot_collision_free(robot_center=coordinate,
                                        robot_radius=robot_radius):
            return coordinate

        return None

    
    def get_valid_moves(self, 
                        robot_center: Coordinate, 
                        robot_radius: float) -> List[Coordinate]:
        bounding_indices = self._get_robot_bounding_indices(robot_center, 
                                                            robot_radius)

        x_lower_bound = bounding_indices.lower_x - 1
        y_lower_bound = bounding_indices.lower_y - 1
        x_upper_bound = bounding_indices.upper_x + 1
        y_upper_bound = bounding_indices.upper_y + 1

        valid_moves = []
        for index_y in range(max(0, y_lower_bound), min(self.height - 1, y_upper_bound) + 1):
            low_cell_index = GridIndex(x_lower_bound, index_y)
            low_coordinate = self._generate_potential_move(robot_center, 
                                                           robot_radius, 
                                                           low_cell_index)

            if low_coordinate is not None:
                valid_moves.append(low_coordinate)

            up_cell_index = GridIndex(x_upper_bound, index_y)
            up_coordinate = self._generate_potential_move(robot_center, 
                                                          robot_radius, 
                                                          up_cell_index)

            if up_coordinate is not None:
                valid_moves.append(up_coordinate)

        for index_x in range(max(0, x_lower_bound), min(self.width - 1, x_upper_bound) + 1):
            low_cell_index = GridIndex(index_x, y_lower_bound)
            low_coordinate = self._generate_potential_move(robot_center, 
                                                           robot_radius, 
                                                           low_cell_index)

            if low_coordinate is not None:
                valid_moves.append(low_coordinate)

            up_cell_index = GridIndex(index_x, y_upper_bound)
            up_coordinate = self._generate_potential_move(robot_center, 
                                                          robot_radius, 
                                                          up_cell_index)

            if up_coordinate is not None:
                valid_moves.append(up_coordinate)


        seen = set()
        unique = []
        for move in valid_moves:
            key = (round(move[0], 4), round(move[1], 4))
            if key not in seen:
                seen.add(key)
                unique.append(move)
        return unique
    
    def get_occupied_cells_for_partial_move(self, 
                                          robot_start_pos: Coordinate, 
                                          robot_radius: float,
                                          pos_change: PosChange) -> Set[GridIndex]:
        robot_end_pos = Coordinate(robot_start_pos.x + pos_change.dev_x, 
                                   robot_start_pos.y + pos_change.dev_y)
        return set(self.get_occupied_cells_for_robot(robot_end_pos, robot_radius))
    
    def _generate_reservation_list(self,
                                      start_pos: Coordinate, 
                                      robot_radius: float, 
                                      robot_velocity: float,
                                      current_time: float,
                                      abs_num_major_steps: int,
                                      abs_num_minor_steps: int,
                                      sign_num_major_steps: int,
                                      sign_num_minor_steps: int,
                                      major_y: bool) -> List[MotionReservation]:
        reservations: List[MotionReservation] = []

        scale_ratio = float(abs_num_minor_steps) / float(abs_num_major_steps)
                
        for i in range(abs_num_major_steps):
            prev_minor_displacement = sign_num_minor_steps * scale_ratio * i * self.cell_size
            prev_major_displacement = sign_num_major_steps * i * self.cell_size
            if major_y:
                prev_pos_change = PosChange(dev_x=prev_minor_displacement, dev_y=prev_major_displacement)
            else:
                prev_pos_change = PosChange(dev_x=prev_major_displacement, dev_y=prev_minor_displacement)

            prev_cells: Set[GridIndex] = self.get_occupied_cells_for_partial_move(start_pos=start_pos, 
                                                                                  robot_radius=robot_radius, 
                                                                                  pos_change=prev_pos_change)

            prev_total_displacement = np.sqrt(prev_minor_displacement**2 + prev_major_displacement**2)
            prev_time_to_end = prev_total_displacement / robot_velocity if robot_velocity > 0 else 0

            minor_displacement = sign_num_minor_steps * scale_ratio * (i+1) * self.cell_size
            major_displacement = sign_num_major_steps * (i+1) * self.cell_size
            if major_y:
                pos_change = PosChange(dev_x=minor_displacement, dev_y=major_displacement)
            else:
                pos_change = PosChange(dev_x=major_displacement, dev_y=minor_displacement)
            curr_cells: Set[GridIndex] = self.get_occupied_cells_for_partial_move(start_pos=start_pos, 
                                                                  robot_radius=robot_radius, 
                                                                  pos_change=pos_change)
            total_displacement = np.sqrt(minor_displacement**2 + major_displacement**2)
            time_to_end = total_displacement / robot_velocity if robot_velocity > 0 else 0
            
            union_cells = prev_cells.union(curr_cells)
            time_interval = TimeInterval(start=current_time + prev_time_to_end, 
                                         end=current_time + time_to_end)
            robot_occupancy = RobotOccupancy(occupied_cells=union_cells,
                                             robot_center=Coordinate(start_pos.x + pos_change.dev_x, 
                                                                     start_pos.y + pos_change.dev_y),
                                             robot_radius=robot_radius)
            motion_reservation =  MotionReservation(time_interval=time_interval,
                                                    robot_occupancy=robot_occupancy)
            reservations.append(motion_reservation)
        return reservations

    def get_reservations_for_move(self, 
                                  robot_start_pos: Coordinate, 
                                  robot_end_pos: Coordinate,
                                  robot_radius: float, 
                                  robot_velocity: float, 
                                  current_time: float) -> List[MotionReservation]:
        
        num_y_steps = round((robot_end_pos.y - robot_start_pos.y) / self.cell_size)
        num_x_steps = round((robot_end_pos.x - robot_start_pos.x) / self.cell_size)
        print(f"cell_size: {self.cell_size}")
        print(f"y_diff: {robot_end_pos.y - robot_start_pos.y}")
        print(f"x_diff: {robot_end_pos.x - robot_start_pos.x}")
        print(f"num_x_steps: {num_x_steps}")
        print(f"num_y_steps: {num_y_steps}")

        if num_y_steps == 0 and num_x_steps == 0:
            time_interval = TimeInterval(start=current_time, 
                                         end=current_time)
            robot_occupancy = RobotOccupancy(occupied_cells=set(self.get_occupied_cells_for_robot(
                                                                robot_center=robot_start_pos,
                                                                robot_radius=robot_radius)),
                                         robot_center=robot_start_pos,
                                         robot_radius=robot_radius)
            reservations = [MotionReservation(time_interval=time_interval,
                                              robot_occupancy=robot_occupancy)]

        else:
            abs_num_x_steps = abs(num_x_steps)
            abs_num_y_steps = abs(num_y_steps)
            sign_num_x_steps = 1 if num_x_steps > 0 else -1
            sign_num_y_steps = 1 if num_y_steps > 0 else -1

            if abs_num_y_steps > abs_num_x_steps:
                major_y = True
                reservations = self._generate_reservation_list(start_pos=robot_start_pos, 
                                                                robot_radius=robot_radius, 
                                                                robot_velocity=robot_velocity,
                                                                current_time=current_time,
                                                                abs_num_major_steps=abs_num_y_steps,
                                                                abs_num_minor_steps=abs_num_x_steps,
                                                                sign_num_major_steps=sign_num_y_steps,
                                                                sign_num_minor_steps=sign_num_x_steps,
                                                                major_y=major_y)
            else:
                major_y = False
                reservations = self._generate_reservation_list(start_pos=robot_start_pos, 
                                                                robot_radius=robot_radius, 
                                                                robot_velocity=robot_velocity,
                                                                current_time=current_time,
                                                                abs_num_major_steps=abs_num_x_steps,
                                                                abs_num_minor_steps=abs_num_y_steps,
                                                                sign_num_major_steps=sign_num_x_steps,
                                                                sign_num_minor_steps=sign_num_y_steps,
                                                                major_y=major_y)

        return reservations
    
    def plot_next_move(self, cell_size: float = 0.03534*2, robot_radius: float = 0.20,
                       start: Tuple[float, float] = (30*0.03534*2, 10*0.03534*2),
                       end: Tuple[float, float] = (32*0.03534*2, 10*0.03534*2), next_positions: List[Tuple[float, float]] = []):
        fig, ax = plt.subplots(figsize=(8, 6))
        for x in np.arange(0, (self.width + 1) * cell_size, cell_size):
            ax.axvline(x, linewidth=0.3)
        for y in np.arange(0, (self.height + 1) * cell_size, cell_size):
            ax.axhline(y, linewidth=0.3)

        for (ix, iy) in self.get_occupied_cells_for_robot(start[0], start[1], robot_radius):
            x0, y0, _, _ = self.cell_rect(ix, iy)
            ax.add_patch(Rectangle((x0, y0), cell_size, cell_size, fill=False, linewidth=1.2))

        # Plot original robot position
        ax.add_patch(Circle(start, robot_radius, fill=False, color='blue', linewidth=2, label='Original Position'))

        # Plot new robot position
        ax.add_patch(Circle(end, robot_radius, fill=False, color='green', linewidth=2, label='New Position'))

        # Plot the line of movement
        ax.plot([start[0], end[0]], [start[1], end[1]], 'r--', linewidth=1, label='Movement')


        preview_radius = robot_radius * 0.05
        for (nx, ny) in next_positions:
            ax.add_patch(Circle((nx, ny), preview_radius, color='red', fill=True))

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, width * cell_size)
        ax.set_ylim(0, height * cell_size)
        ax.set_title("Grid World • Circular Robot Covering Multiple Cells • Potential Moves")
        ax.legend()
        plt.savefig("results/grid_move_example.png")
        plt.close()

    def plot_reservations(self, reservations: Dict[Tuple[float, float], Set[Tuple[int, int]]], cell_size: float,
                          robot_radius: float, start: Tuple[float, float], end: Tuple[float, float]):
        """
        Plots the reserved cells for each time interval.

        Args:
            reservations: A dictionary where keys are time intervals (start_time, end_time)
                        and values are sets of reserved cell coordinates (ix, iy).
            cell_size: The size of each grid cell in meters.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot grid lines
        for x in np.arange(0, (self.width + 1) * cell_size, cell_size):
            ax.axvline(x, linewidth=0.3, color='lightgray')
        for y in np.arange(0, (self.height + 1) * cell_size, cell_size):
            ax.axhline(y, linewidth=0.3, color='lightgray')
        
        # Plot original robot position
        ax.add_patch(Circle(start, robot_radius, fill=False, color='blue', linewidth=1, label='Original Position'))

        # Plot new robot position
        ax.add_patch(Circle(end, robot_radius, fill=False, color='green', linewidth=1, label='New Position'))

        # Plot the line of movement
        ax.plot([start[0], end[0]], [start[1], end[1]], 'r--', linewidth=1, label='Movement')

        # Plot reserved cells with different colors for different time intervals
        color_map = plt.get_cmap('viridis', len(reservations))
        for i, ((start_time, end_time), (center, cells)) in enumerate(reservations.items()):
            print(i)
            color = color_map(i)
            for ix, iy in cells:
                x0, y0, _, _ = self.cell_rect(ix, iy)
                ax.add_patch(Rectangle((x0, y0), cell_size, cell_size, color=color, alpha=0.2, label=f'{start_time:.2f}-{end_time:.2f}'))

            ax.add_patch(Circle(center, robot_radius, fill=False, color=color, linewidth=0.5))

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

        plt.savefig("results/grid_reservations_example.png")
        plt.close()

if __name__ == "__main__":
    width, height = 50, 25
    cell_size = 2*0.03534
    world = GridWorld.empty(width, height, cell_size)

    robot_radius = 0.20
    start = (30*cell_size, 10*cell_size)

    occ_cells = world.get_occupied_cells_for_robot(robot_center_x=start[0], 
                                                   robot_center_y=start[1], 
                                                   robot_radius=robot_radius)
    next_positions = world.get_valid_moves(robot_center_x=start[0], 
                                           robot_center_y=start[1], 
                                           robot_radius=robot_radius)

    #Select a random valid move
    import random
    if next_positions:
        end = random.choice(next_positions)
    else:
        end = start # Stay in place if no valid moves
    
    world.plot_next_move(cell_size=cell_size, 
                         robot_radius=robot_radius, 
                         start=start, 
                         end=end,
                         next_positions=next_positions)
    reservations = world.get_reservations_for_move(start=start, 
                                                   end=end, 
                                                   robot_radius=robot_radius, 
                                                   robot_velocity=0.1, 
                                                   current_time=0.0)
    world.plot_reservations(reservations, cell_size, robot_radius, start, end)

    
        
