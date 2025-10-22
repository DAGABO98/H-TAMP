from typing import List
from matplotlib import pyplot as plt
import numpy as np

from HTAMP.loc_dataclasses import OrientationVector

from HTAMP.traversal_dataclasses import Corridor, Doorway, DriveThrough
from HTAMP.traversal_dataclasses import IntersectionSubgraph
from HTAMP.traversal_dataclasses import DoorwaySubgraph
from HTAMP.traversal_dataclasses import SwitchingPointSubgraph
from HTAMP.traversal_dataclasses import DriveThroughSubgraph


class TraversalGraphPlottingHelper:

    @staticmethod
    def plot_intersection_subgraphs(occupancy_map: np.ndarray, 
                                    origin_x: float, 
                                    origin_y: float, 
                                    resolution: float, 
                                    subgraphs: List[IntersectionSubgraph], 
                                    filename: str):
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
        for subgraph in subgraphs:
            for edge in subgraph.edges:
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                if edge.action == "go_straight":
                    ax.plot(samples_x, samples_y, color='green', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_left":
                    ax.plot(samples_x, samples_y, color='orange', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_right":
                    ax.plot(samples_x, samples_y, color='purple', linewidth=0.5, alpha=0.7)

            for node_label in subgraph.upper_nodes + subgraph.lower_nodes + subgraph.left_nodes + subgraph.right_nodes:
                node = subgraph.nodes_dict[node_label]
                if node.orientation_vec == OrientationVector(1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='blue', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(-1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='cyan', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, 1.0):
                    ax.scatter(node.position.x, node.position.y, color='magenta', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, -1.0):
                    ax.scatter(node.position.x, node.position.y, color='red', s=1, alpha=0.7)

        ax.set_title("Intersection Subgraph Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig(f"{filename}")
        plt.close()

    @staticmethod
    def plot_doorway_subgraphs(occupancy_map: np.ndarray, 
                               origin_x: float, 
                               origin_y: float, 
                               resolution: float,
                               subgraphs: List[DoorwaySubgraph], 
                               filename: str):
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
        for subgraph in subgraphs:
            for edge in subgraph.edges:
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                if edge.action == "go_straight":
                    ax.plot(samples_x, samples_y, color='green', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_left":
                    ax.plot(samples_x, samples_y, color='orange', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_right":
                    ax.plot(samples_x, samples_y, color='purple', linewidth=0.5, alpha=0.7)
                elif edge.action == "switch_directions":
                    ax.plot(samples_x, samples_y, color='brown', linewidth=0.5, alpha=0.7)
    
            for node_label in subgraph.room_nodes + subgraph.doorway_nodes + subgraph.left_nodes + subgraph.right_nodes:
                node = subgraph.nodes_dict[node_label]
                if node.orientation_vec == OrientationVector(1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='blue', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(-1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='cyan', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, 1.0):
                    ax.scatter(node.position.x, node.position.y, color='magenta', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, -1.0):
                    ax.scatter(node.position.x, node.position.y, color='red', s=1, alpha=0.7)

        ax.set_title("Doorway Subgraph Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig(f"{filename}")
        plt.close()
    
    @staticmethod
    def plot_switching_point_subgraphs(occupancy_map: np.ndarray, 
                                       origin_x: float, 
                                       origin_y: float, 
                                       resolution: float,
                                       subgraphs: List[SwitchingPointSubgraph], 
                                       filename: str):
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
        for subgraph in subgraphs:
            for edge in subgraph.edges:
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                if edge.action == "go_straight":
                    ax.plot(samples_x, samples_y, color='green', linewidth=0.5, alpha=0.7)
                elif edge.action == "switch_directions":
                    ax.plot(samples_x, samples_y, color='brown', linewidth=0.5, alpha=0.7)
                elif edge.action == "switch_lanes":
                    ax.plot(samples_x, samples_y, color='orange', linewidth=0.5, alpha=0.7)

            for node_label in subgraph.left_nodes + subgraph.right_nodes:
                node = subgraph.nodes_dict[node_label]
                if node.orientation_vec == (1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='blue', s=1, alpha=0.7)
                elif node.orientation_vec == (-1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='cyan', s=1, alpha=0.7)
                elif node.orientation_vec == (0.0, 1.0):
                    ax.scatter(node.position.x, node.position.y, color='magenta', s=1, alpha=0.7)
                elif node.orientation_vec == (0.0, -1.0):
                    ax.scatter(node.position.x, node.position.y, color='red', s=1, alpha=0.7)

        ax.set_title("Switching Point Subgraph Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig(f"{filename}")
        plt.close()
    
    @staticmethod
    def plot_drive_through_subgraphs(occupancy_map: np.ndarray, 
                                     origin_x: float, 
                                     origin_y: float, 
                                     resolution: float,
                                     subgraphs: List[DriveThroughSubgraph], 
                                     filename: str):
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
        for subgraph in subgraphs:
            for edge in subgraph.edges:
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                if edge.action == "go_straight":
                    ax.plot(samples_x, samples_y, color='green', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_left":
                    ax.plot(samples_x, samples_y, color='orange', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_right":
                    ax.plot(samples_x, samples_y, color='purple', linewidth=0.5, alpha=0.7)

            for node_label in subgraph.entry_nodes + subgraph.exit_nodes + subgraph.left_entry_nodes + subgraph.right_entry_nodes + subgraph.left_exit_nodes + subgraph.right_exit_nodes:
                node = subgraph.nodes_dict[node_label]
                if node.orientation_vec == OrientationVector(1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='blue', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(-1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='cyan', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, 1.0):
                    ax.scatter(node.position.x, node.position.y, color='magenta', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, -1.0):
                    ax.scatter(node.position.x, node.position.y, color='red', s=1, alpha=0.7)

        ax.set_title("Drive-Through Subgraph Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig(f"{filename}")
        plt.close()

    @staticmethod
    def plot_subgraphs_in_one_plot(occupancy_map: np.ndarray, 
                                   origin_x: float, 
                                   origin_y: float, 
                                   resolution: float,
                                   intersection_subgraphs: List[IntersectionSubgraph],
                                   doorway_subgraphs: List[DoorwaySubgraph],
                                   drive_through_subgraphs: List[DriveThroughSubgraph],
                                   filename: str):
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
        for subgraph in intersection_subgraphs:
            for edge in subgraph.edges:
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                if edge.action == "go_straight":
                    ax.plot(samples_x, samples_y, color='green', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_left":
                    ax.plot(samples_x, samples_y, color='orange', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_right":
                    ax.plot(samples_x, samples_y, color='purple', linewidth=0.5, alpha=0.7)

            for node_label in subgraph.upper_nodes + subgraph.lower_nodes + subgraph.left_nodes + subgraph.right_nodes:
                node = subgraph.nodes_dict[node_label]
                if node.orientation_vec == OrientationVector(1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='blue', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(-1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='cyan', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, 1.0):
                    ax.scatter(node.position.x, node.position.y, color='magenta', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, -1.0):
                    ax.scatter(node.position.x, node.position.y, color='red', s=1, alpha=0.7)
                    
        for subgraph in doorway_subgraphs:
            for edge in subgraph.edges:
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                if edge.action == "go_straight":
                    ax.plot(samples_x, samples_y, color='green', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_left":
                    ax.plot(samples_x, samples_y, color='orange', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_right":
                    ax.plot(samples_x, samples_y, color='purple', linewidth=0.5, alpha=0.7)
                elif edge.action == "switch_directions":
                    ax.plot(samples_x, samples_y, color='brown', linewidth=0.5, alpha=0.7)

            for node_label in subgraph.room_nodes + subgraph.doorway_nodes + subgraph.left_nodes + subgraph.right_nodes:
                node = subgraph.nodes_dict[node_label]
                if node.orientation_vec == OrientationVector(1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='blue', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(-1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='cyan', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, 1.0):
                    ax.scatter(node.position.x, node.position.y, color='magenta', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, -1.0):
                    ax.scatter(node.position.x, node.position.y, color='red', s=1, alpha=0.7)

        for subgraph in drive_through_subgraphs:
            for edge in subgraph.edges:
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                if edge.action == "go_straight":
                    ax.plot(samples_x, samples_y, color='green', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_left":
                    ax.plot(samples_x, samples_y, color='orange', linewidth=0.5, alpha=0.7)
                elif edge.action == "turn_right":
                    ax.plot(samples_x, samples_y, color='purple', linewidth=0.5, alpha=0.7)

            for node_label in subgraph.entry_nodes + subgraph.exit_nodes + subgraph.left_entry_nodes + subgraph.right_entry_nodes + subgraph.left_exit_nodes + subgraph.right_exit_nodes:
                node = subgraph.nodes_dict[node_label]
                if node.orientation_vec == OrientationVector(1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='blue', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(-1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='cyan', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, 1.0):
                    ax.scatter(node.position.x, node.position.y, color='magenta', s=1, alpha=0.7)
                elif node.orientation_vec == OrientationVector(0.0, -1.0):
                    ax.scatter(node.position.x, node.position.y, color='red', s=1, alpha=0.7)

        ax.set_title("All Subgraphs Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig(f"{filename}")
        plt.close()


    def plot_extracted_structs(occupancy_map: np.ndarray, 
                               origin_x: float, 
                               origin_y: float, 
                               resolution: float,
                               corridors: List[Corridor],
                               drive_throughs: List[DriveThrough],
                               doorways: List[Doorway],
                               filename: str):
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

        for corridor in corridors:
            for lane in corridor.lanes:
                ax.plot([lane.start_point.x, lane.end_point.x],
                        [lane.start_point.y, lane.end_point.y],
                        color='blue', linewidth=1)
                
        for dt in drive_throughs:
            for lane in dt.lanes:
                ax.plot([lane.start_point.x, lane.end_point.x],
                        [lane.start_point.y, lane.end_point.y],
                        color='green', linewidth=1)
                
        for dw in doorways:
            for lane in dw.lanes:
                ax.scatter([lane.start_point.x, lane.end_point.x],
                        [lane.start_point.y, lane.end_point.y],
                        color='red', s=1)
                
        ax.set_title("Extracted Structs Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig(f"{filename}")
        plt.close()