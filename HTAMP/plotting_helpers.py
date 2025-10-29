from typing import List
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from HTAMP.loc_dataclasses import MotionReservation, OrientationVector

from HTAMP.traversal_dataclasses import Corridor, Doorway, DriveThrough
from HTAMP.traversal_dataclasses import IntersectionSubgraph
from HTAMP.traversal_dataclasses import DoorwaySubgraph
from HTAMP.traversal_dataclasses import SwitchingPointSubgraph
from HTAMP.traversal_dataclasses import DriveThroughSubgraph
from HTAMP.traversal_dataclasses import TraversalGraph


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
                                   switching_point_subgraphs: List[SwitchingPointSubgraph],
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
        
        for subgraph in switching_point_subgraphs:
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
    
    @staticmethod
    def gradient_line(ax: plt.Axes,
                      xy: np.ndarray,
                      color0,
                      color1,
                      linewidth: float = 4.0,
                      alpha: float = 1.0,
                      zorder: int = 2) -> LineCollection:
        """
        Draw polyline 'xy' with a smooth color gradient from color0 -> color1.
        """
        import matplotlib.colors as mcolors

        xy = np.asarray(xy, float)
        if len(xy) < 2:
            raise ValueError("xy needs at least 2 points")

        segs = np.stack([xy[:-1], xy[1:]], axis=1)  # (n-1, 2, 2)
        c0 = np.array(mcolors.to_rgba(color0), float)
        c1 = np.array(mcolors.to_rgba(color1), float)

        nseg = segs.shape[0]
        t = np.linspace(0, 1, nseg)[:, None]       # (nseg,1)
        cols = (1 - t) * c0 + t * c1               # (nseg,4)
        cols[:, 3] *= alpha

        lc = LineCollection(
            segs, colors=cols, linewidths=linewidth,
            zorder=zorder, capstyle="round", joinstyle="round"
        )
        ax.add_collection(lc)
        return lc
    
    def _get_color_for_orientation_vec(self, orientation_vec: OrientationVector) -> str:
        if orientation_vec == OrientationVector(1.0, 0.0):
            return '#2b83ba'
        elif orientation_vec == OrientationVector(-1.0, 0.0):
            return '#fdae61'
        elif orientation_vec == OrientationVector(0.0, 1.0):
            return '#abdda4'
        elif orientation_vec == OrientationVector(0.0, -1.0):
            return '#d7191c'
        else:
            return 'black'
    
    @staticmethod
    def plot_traversal_graph(occupancy_map: np.ndarray, 
                             origin_x: float, 
                             origin_y: float, 
                             resolution: float,
                             traversal_graph: TraversalGraph,
                             filename: str,
                             alpha: float = 0.5):
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

        for edge in traversal_graph.edges:
            samples_x = edge.edge_connector.connector_dict['X']
            samples_y = edge.edge_connector.connector_dict['Y']
            from_node = traversal_graph.nodes_dict[edge.from_node]
            to_node = traversal_graph.nodes_dict[edge.to_node]

            color1 = TraversalGraphPlottingHelper()._get_color_for_orientation_vec(from_node.orientation_vec)
            color2 = TraversalGraphPlottingHelper()._get_color_for_orientation_vec(to_node.orientation_vec)

            TraversalGraphPlottingHelper.gradient_line(ax,
                                                       xy=np.array(list(zip(samples_x, samples_y))),
                                                       color0=color1,
                                                       color1=color2,
                                                       linewidth=0.5,
                                                       alpha=alpha,
                                                       zorder=2)

        for node_label, node in traversal_graph.nodes_dict.items():
            color = TraversalGraphPlottingHelper()._get_color_for_orientation_vec(node.orientation_vec)
            ax.scatter(node.position.x, node.position.y, color=color, s=1, alpha=alpha)

        ax.set_title("Traversal Graph Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig(f"{filename}")
        plt.close()

    @staticmethod
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
    
    @staticmethod
    def plot_motion_reservations(occupancy_map: np.ndarray,
                                    origin_x: float,
                                    origin_y: float,
                                    resolution: float,
                                    motion_reservations: List[MotionReservation],
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
    
            for reservation in motion_reservations:
                occupied_cells = reservation.robot_occupancy.occupied_cells
                cell_xs = [cell.index_x * resolution for cell in occupied_cells]
                cell_ys = [cell.index_y * resolution for cell in occupied_cells]
                ax.scatter(cell_xs, cell_ys, s=0.01, alpha=0.5)
    
            ax.set_title("Motion Reservations Overlay")
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            plt.savefig(f"{filename}")
            plt.close()