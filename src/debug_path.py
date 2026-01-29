"""
Debug path planning to see what path A* actually finds
"""
import cv2
import numpy as np
import os
from segmentation import GeoSAMSegmenter
from map_builder import MapBuilder
from astar import AStarPlanner

print("="*70)
print("PATH PLANNING DEBUG")
print("="*70)

# Load image and segment
image_path = os.path.join("..", "src", "data", "test_images", "default_scene.jpg")
image = cv2.imread(image_path)

segmenter = GeoSAMSegmenter()
segmenter.load_model()

print("\nSegmenting image...")
mask = segmenter.segment(image)

# Build map
print("Building map...")
map_builder = MapBuilder()
map_builder.update_from_segmentation(mask)

# Set start and goal
start = (10, 10)
goal = (90, 90)

print(f"\nStart: {start}")
print(f"Goal: {goal}")

# Check if start and goal are free
print(f"Start is free: {map_builder.is_cell_free(start[0], start[1])}")
print(f"Goal is free: {map_builder.is_cell_free(goal[0], goal[1])}")

# Plan path
print("\nPlanning path with A*...")
planner = AStarPlanner(map_builder)
path = planner.plan(start, goal)

if path:
    print(f"✓ Path found: {len(path)} waypoints")
    
    # Analyze the path
    print("\nPath analysis:")
    print(f"  First 5 waypoints: {path[:5]}")
    print(f"  Last 5 waypoints: {path[-5:]}")
    
    # Calculate path length
    total_distance = 0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        total_distance += np.sqrt(dx**2 + dy**2)
    
    print(f"  Total distance: {total_distance:.1f} cells")
    
    # Straight line distance
    straight_dist = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
    print(f"  Straight line: {straight_dist:.1f} cells")
    print(f"  Detour factor: {total_distance / straight_dist:.2f}x")
    
    if total_distance / straight_dist < 1.1:
        print("\n⚠ WARNING: Path is almost straight!")
        print("  This means obstacles aren't blocking the direct route.")
    
    # Check if path goes through obstacles
    print("\nChecking path for obstacles...")
    obstacles_hit = 0
    for waypoint in path:
        if not map_builder.is_cell_free(waypoint[0], waypoint[1]):
            obstacles_hit += 1
    
    if obstacles_hit > 0:
        print(f"⚠ WARNING: Path goes through {obstacles_hit} obstacle cells!")
    else:
        print("✓ Path avoids all obstacles")
    
    # Visualize
    print("\nCreating visualization...")
    
    # Show the map with path
    vis = map_builder.visualize(path=path, start=start, goal=goal)
    
    # Add info overlay
    cv2.putText(vis, f"Waypoints: {len(path)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, f"Distance: {total_distance:.1f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, f"Detour: {total_distance/straight_dist:.2f}x", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow("Planned Path", vis)
    
    print("\n" + "="*70)
    print("Press any key to close...")
    print("="*70)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else:
    print("✗ No path found!")

print("\n✓ Debug complete!")