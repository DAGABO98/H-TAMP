import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

class MapGenerator:
    def __init__(self, image_path: str, map_vis_path: str, comparison_vis_path: str, factor: int = 1, meters_per_pixel: float = 0.036):
        self.image_path = image_path
        self.map_vis_path = map_vis_path
        self.comparison_vis_path = comparison_vis_path
        self.factor = factor
        self.meters_per_pixel = meters_per_pixel
        try:
            self.image, self.height, self.width = self._read_image()
        except Exception as e:
            print(f"Error reading image: {e}")
        
        self.map = self._detect_free_space()
        self.coarse = self._downsample_occupancy_maxpool()
        self.meters_per_cell = self.meters_per_pixel * self.factor
        print(f"Coarse map shape: {self.coarse.shape}, meters per cell: {self.meters_per_cell:.4f}")

    def _read_image(self):
        image = cv2.imread(self.image_path)
        height, width = image.shape[:2]

        return image, height, width
    
    def _detect_free_space(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        cy0, cy1 = int(self.height*0.30), int(self.height*0.70)
        cx0, cx1 = int(self.width*0.30), int(self.width*0.70)
        center_patch = hsv[cy0:cy1, cx0:cx1].reshape(-1, 3)

        # Use robust percentiles from the center to capture the beige texture range
        lowH, lowS, lowV = np.percentile(center_patch, 5, axis=0)
        hiH,  hiS,  hiV  = np.percentile(center_patch, 95, axis=0)

        # Add a small safety margin; clamp to HSV limits (OpenCV Hue in [0,179])
        padH, padS, padV = 5, 15, 15
        floor_low  = np.array([max(0,   lowH - padH), max(0,   lowS - padS), max(0,   lowV - padV)], dtype=np.uint8)
        floor_high = np.array([min(179, hiH  + padH), min(255, hiS  + padS), min(255, hiV  + padV)], dtype=np.uint8)

        floor_mask = cv2.inRange(hsv, floor_low, floor_high) # 255 on floor, 0 elsewhere

        # Fixed, tight-ish blue range: adjust if your blueprint's blue differs.
        blue_low  = np.array([95, 60, 50], dtype=np.uint8)   # H in [95, 135] ≈ blue; S,V thresholds keep only the colored ink
        blue_high = np.array([135, 255, 255], dtype=np.uint8)
        wall_mask = cv2.inRange(hsv, blue_low, blue_high)    # 255 on walls (exact pixels)

        floor_mask[wall_mask > 0] = 0  # Remove any wall pixels from floor mask

        occ = np.full((self.height, self.width), 0, np.int8)   # free by default
        occ[floor_mask > 0] = 0              # free where beige floor is
        occ[wall_mask  > 0] = 1              # occupied where blue ink is

        return occ
    
    def get_pixel_values_at_row(self, y: int):
        if y < 0 or y >= self.height:
            raise ValueError("y coordinate is out of bounds")
        return self.map[y, :]
    
    def get_pixel_values_at_column(self, x: int):
        if x < 0 or x >= self.width:
            raise ValueError("x coordinate is out of bounds")
        return self.map[:, x]
    
    def _pad_to_multiple(self, factor: int, pad_val: int = 0):
        """Pad bottom/right so height and width are multiples of factor."""
        H, W = self.map.shape
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        if pad_h or pad_w:
            temp_map = np.pad(self.map, ((0, pad_h), (0, pad_w)), constant_values=pad_val)
        else:
            temp_map = self.map.copy()
        return temp_map

    def _downsample_occupancy_maxpool(self) -> np.ndarray:
        """
        Downsample a binary occupancy grid by an integer factor using max-pooling.
        occ: 2D array with 1=occupied, 0=free
        factor: integer spatial reduction (e.g., 2, 4, 8)
        """
        assert self.factor >= 1 and self.factor == int(self.factor)
        if self.factor == 1:
            return self.map.copy()
        a = self._pad_to_multiple(self.factor, pad_val=1)  # pad with occupied to be conservative at edges
        H, W = a.shape
        a = a.reshape(H//self.factor, self.factor, W//self.factor, self.factor)
        # max over each block -> occupied if any pixel occupied
        coarse = a.max(axis=(1,3)).astype(np.uint8)
        return coarse

    def visualize_map(self, points=None):
        vis = np.zeros((self.height, self.width), np.uint8)
        vis[self.map == -1] = 255
        vis[self.map ==  0] = 255
        vis[self.map ==  1] = 0

        if points:
            for point in points:
                plt.plot(point[0], point[1], 'ro', markersize=1)

        plt.figure(figsize=(8,8))
        plt.imshow(vis, cmap='gray', origin="upper")
        plt.savefig(self.map_vis_path if self.map_vis_path else "map_visualization.png")
        plt.close()

    def compare_maps(self):
        plt.figure(figsize=(14,5))

        plt.subplot(1,2,1)
        plt.imshow(self.map, cmap="gray_r", origin="upper")
        plt.title("Original grid")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(self.coarse, cmap="gray_r", origin="upper")
        ttl = f"Downsampled (factor={self.factor}"
        if self.meters_per_cell is not None:
            ttl += f", ~{self.meters_per_cell:.3f} m/cell"
        ttl += ")"
        plt.title(ttl)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(self.comparison_vis_path if self.comparison_vis_path else "map_comparison.png")
        plt.close()
    
    def save_downsampled_map(self, path: str):
        np.save(path, self.coarse)
        print(f"Map saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="maps/FA3/FA3_Top_View.png", help="Path to the input image")
    parser.add_argument("--map_vis_path", type=str, default="results/environment/map_visualization.png", help="Path to save the map visualization")
    parser.add_argument("--comparison_vis_path", type=str, default="results/environment/comparison_visualization.png", help="Path to save the comparison visualization")
    parser.add_argument("--map_path", type=str, default="maps/FA3/occupancy_map.npy", help="Path to save the occupancy map")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image")
    args = parser.parse_args()

    generator = MapGenerator(image_path=args.image_path,
                             map_vis_path=args.map_vis_path,
                             comparison_vis_path=args.comparison_vis_path,
                             factor=args.factor,
                             meters_per_pixel=args.meters_per_pixel)
    points = [(1645, 1060), (1611, 1060), (1700, 982), (1700, 1051)]  # Example points to overlay
    generator.visualize_map(points)
    generator.compare_maps()
    generator.save_downsampled_map(args.map_path)


