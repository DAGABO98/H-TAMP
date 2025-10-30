from typing import Dict, List, Optional, Optional
from matplotlib import pyplot as plt
import numpy as np

class GeometryPlottingHelper:
    
    @staticmethod
    def plot_segments(seg_list: List[dict], 
                      title: str = "Connector split into N equal-length segments", 
                      output_path: str = "curved_connector_segments.png"):
        fig, ax = plt.subplots(figsize=(7,5))
        styles = ["-", "--", "-.", ":", (0, (3,1,1,1))]  # cycles if N>5
        for i, s in enumerate(seg_list):
            ax.plot(s["X"], s["Y"], linewidth=2, linestyle=styles[i % len(styles)], label=f"{i+1}")
        ax.set_aspect('equal', adjustable='box'); ax.grid(True); ax.legend(); ax.set_title(title)
        plt.savefig(output_path)
        plt.close()
    
    @staticmethod
    def plot_connector(connector_dict: Dict[str, np.array],
                       title: Optional[str] = None, 
                       output_path : str ="curved_connector.png"):
        import numpy as np
        A, B = connector_dict["A"], connector_dict["B"]
        X, Y = connector_dict["X"], connector_dict["Y"]

        # Determine a reasonable span for drawing guide lines
        span = max(10.0, 1.5 * max(1.0, np.linalg.norm(B - A)))

        fig, ax = plt.subplots(figsize=(7, 5))

        # The curve itself
        ax.plot(X, Y, linewidth=2)

        # Endpoints
        ax.scatter([A[0], B[0]], [A[1], B[1]], s=50, zorder=3)

        # Guide lines for tangents
        L = connector_dict.get("L", max(np.linalg.norm(connector_dict["T0"]), 
                                             np.linalg.norm(connector_dict["T1"])) + 1e-9)
        ax.arrow(A[0], A[1], connector_dict["T0"][0], connector_dict["T0"][1],
                head_width=0.03*L, head_length=0.06*L, length_includes_head=True)
        ax.arrow(B[0], B[1], connector_dict["T1"][0], connector_dict["T1"][1],
                head_width=0.03*L, head_length=0.06*L, length_includes_head=True)

        ax.set_title(title or f"Connector from ({A[0]:.1f}, {A[1]:.1f}) to ({B[0]:.1f}, {B[1]:.1f})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        plt.savefig(output_path)
        plt.close()