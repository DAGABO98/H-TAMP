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

        fig, ax = plt.subplots(figsize=(8, 8))
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
                ax.plot(samples_x, samples_y, color=colors(i), linewidth=0.5)
                circle = Circle((start_node.position.x, start_node.position.y), robot_profiles[i].radius, color=colors(i), alpha=0.3)
                ax.add_patch(circle)
            # Draw the last position
            end_node, end_interval = path[-1]
            circle = Circle((end_node.position.x, end_node.position.y), robot_profiles[i].radius, color=colors(i), alpha=0.3)
            ax.add_patch(circle)

        plt.savefig("results/motion_planning/planned_paths.svg")
        plt.close()