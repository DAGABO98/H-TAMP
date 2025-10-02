

import argparse
from typing import List, Tuple

from matplotlib import pyplot as plt
import numpy as np
from HTAMP.grid_world import Coordinate

class GeometryHelper:

    @staticmethod
    def unit_vector(vec: Tuple[float, float]) -> Tuple[float, float]:
        new_vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(new_vec)
        if norm == 0:
            return (0.0, 0.0)
        return (new_vec[0]/norm, new_vec[1]/norm)

    @staticmethod
    def cubic_hermite(A: np.ndarray, 
                      B: np.ndarray, 
                      T0: np.ndarray, 
                      T1: np.ndarray, 
                      n: int = 200):
        """
        Evaluate a cubic Hermite curve with endpoints A,B and tangents T0,T1.
        Returns (X, Y) arrays of length n.
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        T0 = np.asarray(T0, dtype=float)
        T1 = np.asarray(T1, dtype=float)

        t = np.linspace(0.0, 1.0, n)
        h00 =  2*t**3 - 3*t**2 + 1
        h10 =      t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 =      t**3 -     t**2

        C = (h00[:,None]*A + h10[:,None]*T0 + h01[:,None]*B + h11[:,None]*T1)
        return C[:,0], C[:,1]
    
    @staticmethod
    def bezier_from_hermite(A: np.ndarray, 
                            B: np.ndarray, 
                            T0: np.ndarray, 
                            T1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P0 = A; P1 = A + T0/3.0; P2 = B - T1/3.0; P3 = B
        return P0, P1, P2, P3

    @staticmethod
    def hermite_from_bezier(P0: np.ndarray, 
                            P1: np.ndarray, 
                            P2: np.ndarray, 
                            P3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = P0; B = P3
        T0 = 3.0*(P1 - P0); T1 = 3.0*(P3 - P2)
        return A, B, T0, T1

    @staticmethod
    def bezier_split(P0: np.ndarray, 
                     P1: np.ndarray, 
                     P2: np.ndarray, 
                     P3: np.ndarray, 
                     t: float) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        Q0 = (1-t)*P0 + t*P1
        Q1 = (1-t)*P1 + t*P2
        Q2 = (1-t)*P2 + t*P3
        R0 = (1-t)*Q0 + t*Q1
        R1 = (1-t)*Q1 + t*Q2
        S  = (1-t)*R0 + t*R1
        L0, L1, L2, L3 = P0, Q0, R0, S
        R0b, R1b, R2b, R3b = S, R1, Q2, P3
        return (L0,L1,L2,L3), (R0b,R1b,R2b,R3b)
    
    @staticmethod
    def subdivide_hermite_at(A: np.ndarray, 
                             B: np.ndarray, 
                             T0: np.ndarray, 
                             T1: np.ndarray, 
                             t: float):
        P0, P1, P2, P3 = GeometryHelper.bezier_from_hermite(A, B, T0, T1)
        (L0,L1,L2,L3), (R0,R1,R2,R3) = GeometryHelper.bezier_split(P0, P1, P2, P3, t)
        return (GeometryHelper.hermite_from_bezier(L0,L1,L2,L3),
                GeometryHelper.hermite_from_bezier(R0,R1,R2,R3))
    
    @staticmethod
    def hermite_d1(A: np.ndarray, 
                   B: np.ndarray, 
                   T0: np.ndarray, 
                   T1: np.ndarray, 
                   t: float):
        t = np.asarray(t, dtype=float)
        h00p =  6*t**2 - 6*t
        h10p =  3*t**2 - 4*t + 1
        h01p = -6*t**2 + 6*t
        h11p =  3*t**2 - 2*t
        dC = (h00p[...,None]*A + h10p[...,None]*T0 +
            h01p[...,None]*B + h11p[...,None]*T1)
        return dC
    
    @staticmethod
    def arc_length_simpson(A: np.ndarray, 
                           B: np.ndarray, 
                           T0: np.ndarray, 
                           T1: np.ndarray, 
                           tol: float = 1e-7,
                           max_depth: int = 20):
        def f(u):
            d = GeometryHelper.hermite_d1(A, B, T0, T1, u)
            return np.linalg.norm(d, axis=-1)
        def simpson(f, a, b):
            c = 0.5*(a+b); fa, fc, fb = f(a), f(c), f(b)
            return (b-a)*(fa + 4*fc + fb)/6.0, fa, fc, fb
        def recurse(a, b, S, fa, fc, fb, depth):
            c = 0.5*(a+b)
            S_l, fa_l, fc_l, fb_l = simpson(f, a, c)
            S_r, fa_r, fc_r, fb_r = simpson(f, c, b)
            if depth <= 0: return S_l + S_r
            if abs(S_l + S_r - S) <= 15*tol:
                return S_l + S_r + (S_l + S_r - S)/15.0
            return (recurse(a, c, S_l, fa_l, fc_l, fb_l, depth-1) +
                    recurse(c, b, S_r, fa_r, fc_r, fb_r, depth-1))
        S0, fa, fc, fb = simpson(f, 0.0, 1.0)
        return recurse(0.0, 1.0, S0, fa, fc, fb, max_depth)

    @staticmethod
    def partial_arc_length_simpson(A: np.ndarray, 
                                   B: np.ndarray, 
                                   T0: np.ndarray, 
                                   T1: np.ndarray, 
                                   t: float, 
                                   tol: float = 1e-8, 
                                   max_depth: int = 20):
        t = float(t)
        if t <= 0: return 0.0
        if t >= 1: return GeometryHelper.arc_length_simpson(A, B, T0, T1, tol=tol, max_depth=max_depth)
        def f(u):
            d = GeometryHelper.hermite_d1(A, B, T0, T1, u)
            return np.linalg.norm(d, axis=-1)
        def simpson_local(f, a, b):
            c = 0.5*(a+b); fa, fc, fb = f(a), f(c), f(b)
            return (b-a)*(fa + 4*fc + fb)/6.0, fa, fc, fb
        def recurse(a, b, S, fa, fc, fb, depth):
            c = 0.5*(a+b)
            S_l, fa_l, fc_l, fb_l = simpson_local(f, a, c)
            S_r, fa_r, fc_r, fb_r = simpson_local(f, c, b)
            if depth <= 0: return S_l + S_r
            if abs(S_l + S_r - S) <= 15*tol:
                return S_l + S_r + (S_l + S_r - S)/15.0
            return (recurse(a, c, S_l, fa_l, fc_l, fb_l, depth-1) +
                    recurse(c, b, S_r, fa_r, fc_r, fb_r, depth-1))
        S0, fa, fc, fb = simpson_local(f, 0.0, t)
        return recurse(0.0, t, S0, fa, fc, fb, max_depth)
    
    @staticmethod
    def find_t_for_length(A: np.ndarray, 
                          B: np.ndarray, 
                          T0: np.ndarray, 
                          T1: np.ndarray, 
                          target_len: float, 
                          tol_len: float = 1e-7):
        L_total = GeometryHelper.arc_length_simpson(A, B, T0, T1, tol=max(tol_len*0.1, 1e-10))
        target_len = float(np.clip(target_len, 0.0, L_total))
        if target_len <= 0: return 0.0
        if target_len >= L_total: return 1.0
        a, b = 0.0, 1.0
        for _ in range(2):
            m = 0.5*(a+b)
            s = GeometryHelper.partial_arc_length_simpson(A, B, T0, T1, m, tol=max(tol_len*0.1, 1e-10))
            a, b = (m, b) if s < target_len else (a, m)
        for _ in range(60):
            t = 0.5*(a+b)
            s = GeometryHelper.partial_arc_length_simpson(A, B, T0, T1, t, tol=max(tol_len*0.1, 1e-10))
            if abs(s - target_len) <= tol_len: return float(t)
            a, b = (t, b) if s < target_len else (a, t)
        return float(0.5*(a+b))
    
    @staticmethod
    def plot_segments(seg_list, 
                      title="Connector split into N equal-length segments", 
                      output_path="curved_connector_segments.png"):
        fig, ax = plt.subplots(figsize=(7,5))
        styles = ["-", "--", "-.", ":", (0, (3,1,1,1))]  # cycles if N>5
        for i, s in enumerate(seg_list):
            ax.plot(s["X"], s["Y"], linewidth=2, linestyle=styles[i % len(styles)], label=f"{i+1}")
        ax.set_aspect('equal', adjustable='box'); ax.grid(True); ax.legend(); ax.set_title(title)
        plt.savefig(output_path)
        plt.close()


class CurvedConnector:
    def __init__(self,
                 origin: Coordinate, 
                 destination: Coordinate, 
                 vec_origin: Tuple[float, float], 
                 vec_destination: Tuple[float, float],
                 tangent_scaling_factor: float = 1.0,
                 num_samples: int = 400):
        self.origin = origin
        self.destination = destination
        self.vec_origin = vec_origin
        self.vec_destination = vec_destination
        self.connector_dict = self._generate_connector_dict(tangent_scaling_factor=tangent_scaling_factor, 
                                                            num_samples=num_samples)

    def _generate_connector_dict(self, tangent_scaling_factor: float, num_samples: int) -> dict[str, np.array]:
        # Generate points along the curved connector
        unit_vec_origin = GeometryHelper.unit_vector(self.vec_origin)
        unit_vec_destination = GeometryHelper.unit_vector(self.vec_destination)

        A = np.asarray([self.origin.x, self.origin.y], dtype=float)
        B = np.asarray([self.destination.x, self.destination.y], dtype=float)

        # Scale tangents by distance between points
        distance = np.linalg.norm(np.array([self.destination.x - self.origin.x, 
                                            self.destination.y - self.origin.y]))
        L = distance * tangent_scaling_factor  # Arbitrary scaling factor for tangents

        T0 = (unit_vec_origin[0] * L, unit_vec_origin[1] * L)
        T1 = (unit_vec_destination[0] * L, unit_vec_destination[1] * L)
        T0 = np.asarray(T0, dtype=float)
        T1 = np.asarray(T1, dtype=float)

        x_points, y_points = GeometryHelper.cubic_hermite(A, B, T0, T1, n=num_samples)
        return {"X": x_points, "Y": y_points,
                "A": A, "B": B, "T0": T0, "T1": T1,
                "L": L}
    
    def _conn_from_hermite_piece(self, A, B, T0, T1, n=400):
        X, Y = GeometryHelper.cubic_hermite(A, B, T0, T1, n=n)
        out = {
            "X": X, "Y": Y,
            "A": A, "B": B, "T0": T0, "T1": T1,
            "L": self.connector_dict.get("L", max(np.linalg.norm(T0), np.linalg.norm(T1)))
        }
        return out

    def split_connector_into_n(self, n_segments=5, tol_len=1e-7, n_samples=400):
        """
        Split a connector dict into n_segments equal-arc pieces.
        Returns: list_of_seg_conns, info_dict
        info_dict has keys: 't_list' (global params), 'L_total', 'L_target'
        """
        assert n_segments >= 2, "n_segments must be >= 2"
        A, B, T0, T1 = self.connector_dict["A"], self.connector_dict["B"], self.connector_dict["T0"], self.connector_dict["T1"]

        L_total = GeometryHelper.arc_length_simpson(A, B, T0, T1, tol=max(tol_len*0.1, 1e-10))
        L_target = L_total / n_segments

        # 1) Find global cut parameters t1..t_{n-1}
        t_targets = [GeometryHelper.find_t_for_length(A, B, T0, T1, i*L_target, tol_len=tol_len)
                    for i in range(1, n_segments)]
        # 2) Perform exact subdivision sequentially with parameter remapping
        segments = []
        left = (A, B, T0, T1)
        t_prev = 0.0
        for i, t_global in enumerate(t_targets, start=1):
            # map global t to local t' on the current 'left'→'right' pair
            t_local = (t_global - t_prev) / max(1e-12, (1.0 - t_prev))
            seg_left, right = GeometryHelper.subdivide_hermite_at(*left, t_local)
            seg_conn = self._conn_from_hermite_piece(*seg_left, n=n_samples)
            segments.append(seg_conn)
            left = right
            t_prev = t_global
        # final piece
        seg_conn = self._conn_from_hermite_piece(*left, n=n_samples)
        segments.append(seg_conn)

        info = {"t_list": t_targets, "L_total": L_total, "L_target": L_target}
        return segments, info

    def plot_connector(self, title=None, output_path="curved_connector.png"):
        import numpy as np
        A, B = self.connector_dict["A"], self.connector_dict["B"]
        X, Y = self.connector_dict["X"], self.connector_dict["Y"]

        # Determine a reasonable span for drawing guide lines
        span = max(10.0, 1.5 * max(1.0, np.linalg.norm(B - A)))

        fig, ax = plt.subplots(figsize=(7, 5))

        # The curve itself
        ax.plot(X, Y, linewidth=2)

        # Endpoints
        ax.scatter([A[0], B[0]], [A[1], B[1]], s=50, zorder=3)

        # Guide lines for tangents
        L = self.connector_dict.get("L", max(np.linalg.norm(self.connector_dict["T0"]), 
                                             np.linalg.norm(self.connector_dict["T1"])) + 1e-9)
        ax.arrow(A[0], A[1], self.connector_dict["T0"][0], self.connector_dict["T0"][1],
                head_width=0.03*L, head_length=0.06*L, length_includes_head=True)
        ax.arrow(B[0], B[1], self.connector_dict["T1"][0], self.connector_dict["T1"][1],
                head_width=0.03*L, head_length=0.06*L, length_includes_head=True)

        ax.set_title(title or f"Connector ({self.connector_dict.get('flow','')})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":

    origin = Coordinate(x=0.0, y=0.0)     # point on line 1
    destination1 = Coordinate(x=0.0, y=2.0)     # point on parallel line 2
    destination2 = Coordinate(x=3.0, y=4.0)    # point on parallel line 3 (opposite direction)

    # Traffic runs along +x for the "bottom" line; lines are parallel, so dir_vec is along the lines.
    dir_vec1 = np.array([1.0, 0.0])
    dir_vec2 = np.array([1.0, 0.0])
    dir_vec3 = np.array([-1.0, 0.0])
    dir_vec4 = np.array([0.0, 1.0])

    # Handle factor k: L = k * H   (H is the perpendicular spacing between the lines)
    tangent_scaling_factor = 1.2
    num_samples = 400

    curved_connector_same_dir = CurvedConnector(origin=origin, 
                                                destination=destination1, 
                                                vec_origin=dir_vec1, 
                                                vec_destination=dir_vec2, 
                                                tangent_scaling_factor=tangent_scaling_factor,
                                                num_samples=num_samples)
    curved_connector_same_dir.plot_connector(title="Curved Connector: Same Direction",
                                             output_path="results/curved_connector_same.png")

    curved_connector_opp_dir = CurvedConnector(origin=origin,
                                                destination=destination1,
                                                vec_origin=dir_vec1,
                                                vec_destination=dir_vec3,
                                                tangent_scaling_factor=tangent_scaling_factor,
                                                num_samples=num_samples)
    curved_connector_opp_dir.plot_connector(title="Curved Connector: Opposite Direction",
                                             output_path="results/curved_connector_opp.png")
    
    curved_connector_perp_dir = CurvedConnector(origin=origin,
                                                destination=destination2,
                                                vec_origin=dir_vec1,
                                                vec_destination=dir_vec4,
                                                tangent_scaling_factor=tangent_scaling_factor,
                                                num_samples=num_samples)
    curved_connector_perp_dir.plot_connector(title="Curved Connector: Perpendicular Direction",
                                             output_path="results/curved_connector_perp.png")

    segments, info = curved_connector_perp_dir.split_connector_into_n(n_segments=10, tol_len=1e-8, n_samples=400)
    print(f"t_list: {[f'{t:.9f}' for t in info['t_list']]}")
    print(f"Total length ≈ {info['L_total']:.9f}; each target ≈ {info['L_target']:.9f}")

    # Verify numerically
    lens = [GeometryHelper.arc_length_simpson(s["A"], s["B"], s["T0"], s["T1"], tol=1e-9) for s in segments]
    print("Segment lengths:", [f"{L:.9f}" for L in lens])
    GeometryHelper.plot_segments(segments, 
                                 title="Curved Connector Split into 10 Equal-Length Segments", 
                                 output_path="results/curved_connector_perp_split10.png")
