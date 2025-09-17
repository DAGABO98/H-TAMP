import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Any
from matplotlib.patches import Rectangle, Circle

@dataclass
class Coordinates:
    x: float
    y: float

@dataclass
class Interval:
    start: float
    end: float

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
    
    def is_in_bounds_cell(self, index_x: int, index_y: int) -> bool:
        return 0 <= index_x < self.width and 0 <= index_y < self.height
    
    def cell_rect(self, index_x: int, index_y: int) -> Tuple[float, float, float, float]:
        lower_x = index_x * self.cell_size
        lower_y = index_y * self.cell_size
        return lower_x, lower_y, lower_x + self.cell_size, lower_y + self.cell_size
    
    def robot_intersects_cell(self, robot_center_x: float, robot_center_y: float, 
                              robot_radius: float, index_x: int, index_y: int) -> bool:
        lower_x, lower_y, upper_x, upper_y = self.cell_rect(index_x, index_y)
        selected_x = min(max(robot_center_x, lower_x), upper_x)
        selected_y = min(max(robot_center_y, lower_y), upper_y)
        dev_x = robot_center_x - selected_x
        dev_y = robot_center_y - selected_y
        return ((dev_x * dev_x) + (dev_y * dev_y)) <= (robot_radius * robot_radius)
    
    def _get_robot_bounding_indices(self, 
                                    robot_center_x: float, 
                                    robot_center_y: float,
                                    robot_radius: float):
        lower_index_x = int(np.floor((robot_center_x - robot_radius) / self.cell_size))
        lower_index_y = int(np.floor((robot_center_y - robot_radius) / self.cell_size))
        upper_index_x = int(np.floor((robot_center_x + robot_radius) / self.cell_size))
        upper_index_y = int(np.floor((robot_center_y + robot_radius) / self.cell_size))

        return lower_index_x, lower_index_y, upper_index_x, upper_index_y
    
    def is_robot_in_bounds(self, 
                          robot_center_x: float, 
                          robot_center_y: float,
                          robot_radius: float) -> bool:
        lower_index_x, lower_index_y, upper_index_x, upper_index_y = self._get_robot_bounding_indices(
            robot_center_x, robot_center_y, robot_radius)

        return (0 <= lower_index_x < self.width and
                0 <= lower_index_y < self.height and
                0 <= upper_index_x < self.width and
                0 <= upper_index_y < self.height)

    def get_occupied_cells_for_robot(self, 
                                     robot_center_x: float, 
                                     robot_center_y: float,
                                     robot_radius: float) -> List[Tuple[int, int]]:
        lower_index_x, lower_index_y, upper_index_x, upper_index_y = self._get_robot_bounding_indices(
            robot_center_x, robot_center_y, robot_radius)
        
        occupied_cells = []
        for index_x in range(lower_index_x, upper_index_x + 1):
            for index_y in range(lower_index_y, upper_index_y + 1):
                if self.is_in_bounds_cell(index_x, index_y):
                    if self.robot_intersects_cell(robot_center_x, robot_center_y, robot_radius, index_x, index_y):
                        occupied_cells.append((index_x, index_y))
        return occupied_cells
    
    def is_robot_collision_free(self, 
                                robot_center_x: float, 
                                robot_center_y: float,
                                robot_radius: float) -> bool:
        if not self.is_robot_in_bounds(robot_center_x, robot_center_y, robot_radius):
            return False
        
        occupied_cells = self.get_occupied_cells_for_robot(robot_center_x, robot_center_y, robot_radius)
        for index_x, index_y in occupied_cells:
            if self.occupancy_map[index_y, index_x] == 1:
                return False
        return True
    
    def get_valid_moves(self, 
                        robot_center_x: float, 
                        robot_center_y: float,
                        robot_radius: float) -> List[Tuple[float, float]]:
        robot_bounding_indices = self._get_robot_bounding_indices(robot_center_x, 
                                                                  robot_center_y, 
                                                                  robot_radius)
        
        lower_index_x, lower_index_y, upper_index_x, upper_index_y = robot_bounding_indices

        x_lower_bound = lower_index_x-1
        y_lower_bound = lower_index_y-1
        x_upper_bound = upper_index_x+1
        y_upper_bound = upper_index_y+1

        valid_moves = []
        for index_y in range(max(0, y_lower_bound), min(self.height - 1, y_upper_bound) + 1):
            lower_x0, lower_y0, lower_x1, lower_y1 = self.cell_rect(x_lower_bound, index_y)
            lower_selected_x = min(max(robot_center_x, lower_x0), lower_x1)
            lower_selected_y = min(max(robot_center_y, lower_y0), lower_y1)

            if self.is_robot_collision_free(robot_center_x=lower_selected_x,
                                           robot_center_y=lower_selected_y,
                                           robot_radius=robot_radius):
                valid_moves.append((lower_selected_x, lower_selected_y))

            upper_x0, upper_y0, upper_x1, upper_y1 = self.cell_rect(x_upper_bound, index_y)
            upper_selected_x = min(max(robot_center_x, upper_x0), upper_x1)
            upper_selected_y = min(max(robot_center_y, upper_y0), upper_y1)
            valid_moves.append((upper_selected_x, upper_selected_y))


        for index_x in range(max(0, x_lower_bound), min(self.width - 1, x_upper_bound) + 1):
            lower_x0, lower_y0, lower_x1, lower_y1 = self.cell_rect(index_x, y_lower_bound)
            lower_selected_x = min(max(robot_center_x, lower_x0), lower_x1)
            lower_selected_y = min(max(robot_center_y, lower_y0), lower_y1)
            valid_moves.append((lower_selected_x, lower_selected_y))

            upper_x0, upper_y0, upper_x1, upper_y1 = self.cell_rect(index_x, y_upper_bound)
            upper_selected_x = min(max(robot_center_x, upper_x0), upper_x1)
            upper_selected_y = min(max(robot_center_y, upper_y0), upper_y1)
            valid_moves.append((upper_selected_x, upper_selected_y))


        seen = set()
        unique = []
        for move in valid_moves:
            key = (round(move[0], 4), round(move[1], 4))
            if key not in seen:
                seen.add(key)
                unique.append(move)
        return unique
    
    def get_occupied_cells_for_partial_move(self, 
                                          start_x: float, 
                                          start_y: float,
                                          robot_radius: float,
                                          dev_x: float,
                                          dev_y: float) -> Set[Tuple[int, int]]:
        
        return set(self.get_occupied_cells_for_robot(start_x + dev_x, start_y + dev_y, robot_radius))
    
    def get_reservations_for_move(self, 
                                  start: Tuple[float, float], 
                                  end:Tuple[float, float],
                                  robot_radius: float, 
                                  robot_velocity: float, 
                                  current_time: float) -> Dict[Tuple[float, float], Tuple[Tuple[float, float], Set[Tuple[int, int]]]]:
        reservations: Dict[Tuple[float, float], list[Tuple[int, int]]] = {}
        num_y_steps = round((end[1] - start[1]) / self.cell_size)
        num_x_steps = round((end[0] - start[0]) / self.cell_size)
        print(f"cell_size: {self.cell_size}")
        print(f"y_diff: {end[1] - start[1]}")
        print(f"x_diff: {end[0] - start[0]}")
        print(f"num_x_steps: {num_x_steps}")
        print(f"num_y_steps: {num_y_steps}")

        if num_y_steps == 0 and num_x_steps == 0:
            reservations[(current_time, current_time)] = ((start[0], start[1]), self.get_occupied_cells_for_robot(robot_center_x=start[0], 
                                                                                                                  robot_center_y=start[1], 
                                                                                                                  robot_radius=robot_radius))
        else:
            abs_num_x_steps = abs(num_x_steps)
            abs_num_y_steps = abs(num_y_steps)
            sign_num_x_steps = 1 if num_x_steps > 0 else -1
            sign_num_y_steps = 1 if num_y_steps > 0 else -1

            if abs_num_y_steps > abs_num_x_steps:
                y_ratio = float(abs_num_x_steps) / float(abs_num_y_steps)
                
                for i in range(abs_num_y_steps):
                    prev_x_displacement = sign_num_x_steps * y_ratio * i * self.cell_size
                    prev_y_displacement = sign_num_y_steps * i * self.cell_size
                    prev_cells = self.get_occupied_cells_for_partial_move(start_x=start[0], 
                                                                          start_y=start[1], 
                                                                          robot_radius=robot_radius, 
                                                                          dev_x=prev_x_displacement, 
                                                                          dev_y=prev_y_displacement)
                    prev_total_displacement = np.sqrt(prev_x_displacement**2 + prev_y_displacement**2)
                    prev_time_to_end = prev_total_displacement / robot_velocity if robot_velocity > 0 else 0

                    x_displacement = sign_num_x_steps * y_ratio * (i+1) * self.cell_size
                    y_displacement = sign_num_y_steps * (i+1) * self.cell_size
                    curr_cells = self.get_occupied_cells_for_partial_move(start_x=start[0], 
                                                                          start_y=start[1], 
                                                                          robot_radius=robot_radius, 
                                                                          dev_x=x_displacement, 
                                                                          dev_y=y_displacement)
                    total_displacement = np.sqrt(x_displacement**2 + y_displacement**2)
                    time_to_end = total_displacement / robot_velocity if robot_velocity > 0 else 0
                    
                    union_cells = prev_cells.union(curr_cells)
                    reservations[(current_time + prev_time_to_end, current_time + time_to_end)] = ((start[0]+x_displacement, start[1]+y_displacement), union_cells)
                    
            else:
                x_ratio = float(abs_num_y_steps) / float(abs_num_x_steps)
                for i in range(abs_num_x_steps):
                    prev_x_displacement = sign_num_x_steps * i * self.cell_size
                    prev_y_displacement = sign_num_y_steps * x_ratio * i * self.cell_size
                    prev_cells = self.get_occupied_cells_for_partial_move(start_x=start[0], 
                                                                          start_y=start[1], 
                                                                          robot_radius=robot_radius, 
                                                                          dev_x=prev_x_displacement, 
                                                                          dev_y=prev_y_displacement)
                    prev_total_displacement = np.sqrt(prev_x_displacement**2 + prev_y_displacement**2)
                    prev_time_to_end = prev_total_displacement / robot_velocity if robot_velocity > 0 else 0

                    x_displacement = sign_num_x_steps * (i+1) * self.cell_size
                    y_displacement = sign_num_y_steps * x_ratio * (i+1) * self.cell_size
                    curr_cells = self.get_occupied_cells_for_partial_move(start_x=start[0], 
                                                                          start_y=start[1], 
                                                                          robot_radius=robot_radius, 
                                                                          dev_x=x_displacement, 
                                                                          dev_y=y_displacement)
                    total_displacement = np.sqrt(x_displacement**2 + y_displacement**2)
                    time_to_end = total_displacement / robot_velocity if robot_velocity > 0 else 0

                    union_cells = prev_cells.union(curr_cells)
                    reservations[(current_time + prev_time_to_end, current_time + time_to_end)] = ((start[0]+x_displacement, start[1]+y_displacement), union_cells)

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

    
        
