"""
Debug the entire pipeline step by step
"""
import cv2
import numpy as np
import config
from drone_input import DroneInput
from segmentation import GeoSAMSegmenter
from map_builder import MapBuilder

print("="*70)
print("DEBUGGING THE PIPELINE")
print("="*70)

# Step 1: Get frame
print("\n[1] Getting frame from drone input...")
drone = DroneInput(use_drone=False)
drone.connect()
drone.start_stream()

import time
time.sleep(1)

frame = drone.get_frame()
print(f"Frame shape: {frame.shape}")
print(f"Frame has variation: {frame.std():.2f}")

cv2.imshow("1. Raw Frame", frame)
cv2.waitKey(0)

# Step 2: Segment
print("\n[2] Segmenting frame...")
segmenter = GeoSAMSegmenter()
segmenter.load_model()

mask = segmenter.segment(frame)
print(f"Mask shape: {mask.shape}")
print(f"Unique values in mask: {np.unique(mask)}")
print(f"Safe pixels: {np.sum(mask == 1)} ({np.sum(mask == 1)/mask.size*100:.1f}%)")
print(f"Unsafe pixels: {np.sum(mask == 0)} ({np.sum(mask == 0)/mask.size*100:.1f}%)")

# Show mask
mask_vis = mask.copy() * 255
cv2.imshow("2. Segmentation Mask", mask_vis)

# Show overlay
overlay = segmenter.visualize_segmentation(frame, mask)
cv2.imshow("3. Segmentation Overlay", overlay)
cv2.waitKey(0)

# Step 3: Build map
print("\n[3] Building map...")
map_builder = MapBuilder()
map_builder.update_from_segmentation(mask)

grid = map_builder.get_grid()
print(f"Grid shape: {grid.shape}")
print(f"Unique values in grid: {np.unique(grid)}")
print(f"Free cells: {np.sum(grid == 1)} ({np.sum(grid == 1)/grid.size*100:.1f}%)")
print(f"Obstacle cells: {np.sum(grid == 0)} ({np.sum(grid == 0)/grid.size*100:.1f}%)")

# Show grid
grid_vis = grid.copy() * 255
cv2.imshow("4. Navigation Grid (Before Visualization)", grid_vis)
cv2.waitKey(0)

# Step 4: Visualize map
print("\n[4] Creating map visualization...")
start = (10, 10)
goal = (90, 90)

map_vis = map_builder.visualize(start=start, goal=goal)
cv2.imshow("5. Final Map Visualization", map_vis)
print("\nThis is what you see in main.py!")
cv2.waitKey(0)

cv2.destroyAllWindows()
drone.disconnect()

print("\n" + "="*70)
print("Debug complete! Did you see obstacles in all windows?")
print("="*70)