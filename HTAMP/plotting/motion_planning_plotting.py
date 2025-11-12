from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import numpy as np

from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from HTAMP.environment.loc_dataclasses import Coordinate
from HTAMP.environment.grid_world import TimeInterval, RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalGraph, TraversalNode

class MotionPlanningPlotter:

    @staticmethod
    def plot_paths(occupancy_map: np.ndarray, 
                   origin_x: float, 
                   origin_y: float, 
                   resolution: float, 
                   paths: List[List[Tuple[TraversalNode, TimeInterval]]], 
                   traversal_graph: TraversalGraph,
                   robot_profiles: List[RobotProfile]) -> None:
        
        rows, cols = occupancy_map.shape
        xmin, xmax = origin_x, origin_x + cols * resolution
        ymin, ymax = origin_y, origin_y + rows * resolution

        fig, ax = plt.subplots(figsize=(16, 16), dpi=150)
        im = ax.imshow(
            occupancy_map,
            cmap="gray_r",
            origin="upper",              # flip so (0,0) is top-left
            extent=[xmin, xmax, ymax, ymin],  # still in meters
            aspect="equal"
        )

        # Plot each path
        colors = plt.get_cmap('hsv', len(paths) + 1)
        for i, path in enumerate(paths):
            for j in range(len(path) - 1):
                start_node, start_interval = path[j]
                end_node, end_interval = path[j + 1]
                edge = traversal_graph.edge_dict.get((start_node.label, end_node.label))
                if edge is None:
                    continue
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                ax.plot(samples_x, samples_y, color=colors(i), linewidth=2.0, alpha=0.7)
                circle = Circle((start_node.position.x, start_node.position.y), robot_profiles[i].radius, color=colors(i), alpha=0.3)
                ax.add_patch(circle)
            # Draw the last position
            end_node, end_interval = path[-1]
            circle = Circle((end_node.position.x, end_node.position.y), robot_profiles[i].radius, color=colors(i), alpha=0.3)
            ax.add_patch(circle)

        plt.savefig("results/motion_planning/planned_paths.svg")
        plt.close()
    
    @staticmethod
    def plot_state_debug(occupancy_map: np.ndarray, 
                   origin_x: float, 
                   origin_y: float, 
                   resolution: float, 
                   robot_positions: dict[int, Coordinate], 
                   robots_current_node_index: dict[int, int], 
                   point_indices_on_edge: dict[int, int], 
                   robot_paths: dict[int, list[tuple[TraversalNode, TimeInterval]]], 
                   traversal_graph: TraversalGraph, 
                   robot_profiles: List[RobotProfile], 
                   step_number: int) -> None: 
        rows, cols = occupancy_map.shape 
        xmin, xmax = origin_x, origin_x + cols * resolution 
        ymin, ymax = origin_y, origin_y + rows * resolution 
        fig, ax = plt.subplots(figsize=(16, 16), dpi=150) 
        im = ax.imshow(occupancy_map, 
                       cmap="gray_r", 
                       origin="upper", # flip so (0,0) is top-left 
                       extent=[xmin, xmax, ymax, ymin], # still in meters 
                       aspect="equal" ) 
        colors = plt.get_cmap('hsv', len(robot_paths) + 1) 
        for robot_id in robot_paths.keys(): 
            path = robot_paths[robot_id] 
            robot_position = robot_positions[robot_id] 
            current_node_index = robots_current_node_index[robot_id] 
            print(current_node_index) 
            point_index_on_edge = point_indices_on_edge[robot_id] 
            print(point_index_on_edge) 
            truncated_path = path[current_node_index:] 
            for j in range(len(truncated_path) - 1): 
                start_node, start_interval = truncated_path[j] 
                end_node, end_interval = truncated_path[j + 1] 
                edge = traversal_graph.edge_dict.get((start_node.label, end_node.label)) 
                samples_x = edge.edge_connector.connector_dict['X'] 
                samples_y = edge.edge_connector.connector_dict['Y'] 
                if j == 0: 
                    full_samples_x = samples_x[point_index_on_edge:] 
                    full_samples_y = samples_y[point_index_on_edge:] 
                    ax.plot(full_samples_x, full_samples_y, color=colors(robot_id), linewidth=2.0, alpha=0.7) 
                    circle = Circle((robot_position.x, robot_position.y), robot_profiles[robot_id].radius, color=colors(robot_id)) 
                    ax.add_patch(circle) 
                else:
                    ax.plot(samples_x, samples_y, color=colors(robot_id), linewidth=2.0, alpha=0.7) 
        plt.savefig(f"results/motion_planning/state/state_step_{step_number}.svg") 
        plt.close()
    
    @staticmethod
    # ---- figure setup (done once) ----
    def setup_axes(Occupancy_map: np.ndarray, 
                   origin_x: float, 
                   origin_y: float, 
                   resolution: float) -> Tuple[Figure, Axes, AxesImage]:
        rows, cols = Occupancy_map.shape
        xmin, xmax = origin_x, origin_x + cols * resolution
        ymin, ymax = origin_y, origin_y + rows * resolution
        fig, ax = plt.subplots(figsize=(16, 16), dpi=150)
        # seed with a blank image; we will update it each frame
        im = ax.imshow(Occupancy_map, 
                       cmap="gray_r", 
                       origin="upper",
                       extent=[xmin, xmax, ymax, ymin], 
                       aspect="equal")
        im.set_zorder(0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)  # maintain the flipped y
        ax._dyn = []  # will store per-frame artists (lines, circles) to remove next frame
        return fig, ax, im
    
    @staticmethod
    def _plot_state(ax: Axes,
                   robot_positions: dict[int, Coordinate],
                   robots_current_node_index: dict[int, int],
                   point_indices_on_edge: dict[int, int],
                   robot_paths: dict[int, list[tuple[TraversalNode, TimeInterval]]], 
                   traversal_graph: TraversalGraph,
                   robot_profiles: List[RobotProfile]) -> None:

        # remove dynamic artists from prior frame
        for art in ax._dyn:
            art.remove()
        ax._dyn = []

        # stable color assignment independent of robot_id values
        robot_ids = list(robot_paths.keys())
        colors = plt.get_cmap('hsv', len(robot_ids) + 1)

        for idx, robot_id in enumerate(robot_ids):
            path = robot_paths[robot_id]
            robot_position = robot_positions[robot_id]
            current_node_index = robots_current_node_index[robot_id]
            point_index_on_edge = point_indices_on_edge[robot_id]
            truncated_path = path[current_node_index:]

            if len(truncated_path) < 2:
                circ = Circle((robot_position.x, robot_position.y),
                                robot_profiles[robot_id].radius,
                                color=colors(idx), alpha=0.3)
                ax.add_patch(circ)
                ax._dyn.extend([circ])

            for j in range(len(truncated_path) - 1):
                start_node, _ = truncated_path[j]
                end_node, _ = truncated_path[j + 1]
                edge = traversal_graph.edge_dict.get((start_node.label, end_node.label))
                sx = edge.edge_connector.connector_dict['X']
                sy = edge.edge_connector.connector_dict['Y']

                if j == 0:
                    # from current point along the current edge
                    line, = ax.plot(sx[point_index_on_edge:], sy[point_index_on_edge:],
                                    color=colors(idx), linewidth=2.0, alpha=0.7, animated=True)
                    circ = Circle((robot_position.x, robot_position.y),
                                robot_profiles[robot_id].radius,
                                color=colors(idx))
                    ax.add_patch(circ)
                    ax._dyn.extend([line, circ])
                else:
                    line, = ax.plot(sx, sy, color=colors(idx), linewidth=2.0, alpha=0.7, animated=True)
                    ax._dyn.append(line)
    
    @staticmethod
    def generate_state_animation(occupancy_map: np.ndarray,
                                 origin_x: float,
                                 origin_y: float,
                                 resolution: float,
                                 robot_positions_seq: List[dict[int, Coordinate]],
                                 robots_current_node_index_seq: List[dict[int, int]],
                                 point_indices_on_edge_seq: List[dict[int, int]],
                                 robot_paths_seq: List[dict[int, list[tuple[TraversalNode, TimeInterval]]]],
                                 traversal_graph: TraversalGraph,
                                 robot_profiles: List[RobotProfile],
                                 fps_sim: int,
                                 num_sim_frames: int) -> None:
        fig, ax, im = MotionPlanningPlotter.setup_axes(occupancy_map, origin_x, origin_y, resolution)

        # Use FFmpeg underneath, tuned for speed & compatibility
        writer = FFMpegWriter(
            fps=4*fps_sim,
            codec="libx264",
            bitrate=4000,
            extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast"]
        )

        with writer.saving(fig, "results/motion_planning/sim.mp4", dpi=150):
            for step in range(num_sim_frames):
                print(f"Rendering frame {step+1}/{num_sim_frames}")
                MotionPlanningPlotter._plot_state(ax=ax, 
                                                  robot_positions=robot_positions_seq[step], 
                                                  robots_current_node_index=robots_current_node_index_seq[step],
                                                  point_indices_on_edge=point_indices_on_edge_seq[step], 
                                                  robot_paths=robot_paths_seq[step],
                                                  traversal_graph=traversal_graph, 
                                                  robot_profiles=robot_profiles)
                writer.grab_frame()

        plt.close(fig)
        