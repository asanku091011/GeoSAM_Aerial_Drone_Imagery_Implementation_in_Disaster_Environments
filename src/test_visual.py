"""
Visual testing script - Shows all steps with visualizations
Perfect for understanding what the system is doing!
"""

import cv2
import numpy as np
import time
import config

# Import components
from drone_input import DroneInput
from segmentation import GeoSAMSegmenter
from map_builder import MapBuilder
from astar import AStarPlanner
from rrt_star import RRTStarPlanner
from greedy import GreedyPlanner

def wait_for_key(message="Press any key to continue, 'q' to quit..."):
    """Wait for user to press a key"""
    print(f"\n{message}")
    key = cv2.waitKey(0)
    return key != ord('q')

def main():
    print("\n" + "="*70)
    print("🎨 VISUAL TESTING MODE")
    print("="*70)
    print("\nThis will show you each step of the navigation system")
    print("with detailed visualizations.\n")
    
    # Step 1: Initialize drone input
    print("[Step 1/7] Initializing video input...")
    drone = DroneInput(use_drone=False)
    
    if not drone.connect():
        print("✗ Failed to connect")
        return
    
    drone.start_stream()
    time.sleep(1)
    
    # Get a frame
    frame = drone.get_frame()
    if frame is None:
        print("✗ No frame received")
        return
    
    print("✓ Video input ready")
    cv2.imshow("Step 1: Raw Drone Feed", frame)
    if not wait_for_key():
        return
    
    # Step 2: Segmentation
    print("\n[Step 2/7] Running AI segmentation...")
    segmenter = GeoSAMSegmenter()
    segmenter.load_model()
    
    mask = segmenter.segment(frame)
    stats = segmenter.get_statistics(mask)
    
    print(f"✓ Segmentation complete")
    print(f"  Safe terrain: {stats['safe_percentage']:.1f}%")
    print(f"  Unsafe terrain: {stats['unsafe_percentage']:.1f}%")
    
    # Show segmentation
    seg_vis = segmenter.visualize_segmentation(frame, mask)
    cv2.putText(seg_vis, f"Safe: {stats['safe_percentage']:.1f}%", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Step 2: Segmentation Result", seg_vis)
    
    # Also show just the mask
    mask_vis = mask.copy() * 255
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_vis, "White = Safe, Black = Unsafe", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Step 2b: Binary Mask", mask_vis)
    
    if not wait_for_key():
        return
    
    # Step 3: Build navigation map
    print("\n[Step 3/7] Building navigation map...")
    map_builder = MapBuilder()
    map_builder.update_from_segmentation(mask)
    
    map_stats = map_builder.get_statistics()
    print(f"✓ Map built: {map_stats['width']}x{map_stats['height']} cells")
    print(f"  Free space: {map_stats['free_percentage']:.1f}%")
    
    # Show the map
    map_vis = map_builder.visualize()
    cv2.putText(map_vis, f"Grid: {map_stats['width']}x{map_stats['height']}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Step 3: Navigation Grid", map_vis)
    
    if not wait_for_key():
        return
    
    # Step 4: Set start and goal
    print("\n[Step 4/7] Setting start and goal positions...")
    start = (10, 10)
    goal = (90, 90)
    
    print(f"  Start: {start}")
    print(f"  Goal: {goal}")
    
    # Verify positions are free
    if not map_builder.is_cell_free(start[0], start[1]):
        print("⚠ Warning: Start position is in obstacle!")
    if not map_builder.is_cell_free(goal[0], goal[1]):
        print("⚠ Warning: Goal position is in obstacle!")
    
    # Show start and goal
    sg_vis = map_builder.visualize(start=start, goal=goal)
    cv2.putText(sg_vis, f"Start: {start}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(sg_vis, f"Goal: {goal}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Step 4: Start & Goal", sg_vis)
    
    if not wait_for_key():
        return
    
    # Step 5: Path planning with all three algorithms
    print("\n[Step 5/7] Planning paths with different algorithms...")
    
    algorithms = {
        'A*': AStarPlanner(map_builder),
        'RRT*': RRTStarPlanner(map_builder),
        'Greedy': GreedyPlanner(map_builder)
    }
    
    results = {}
    
    for name, planner in algorithms.items():
        print(f"\n  Testing {name}...")
        start_time = time.time()
        path = planner.plan(start, goal)
        plan_time = time.time() - start_time
        
        if path:
            print(f"  ✓ {name}: {len(path)} waypoints in {plan_time*1000:.1f}ms")
            results[name] = path
            
            # Show this algorithm's path
            algo_vis = map_builder.visualize(path=path, start=start, goal=goal)
            cv2.putText(algo_vis, f"{name}: {len(path)} waypoints", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(algo_vis, f"Time: {plan_time*1000:.0f}ms", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(f"Step 5: {name} Path", algo_vis)
            
            if not wait_for_key(f"{name} complete. Press key for next algorithm..."):
                return
        else:
            print(f"  ✗ {name}: Failed to find path")
    
    # Step 6: Compare all paths
    if results:
        print(f"\n[Step 6/7] Comparing all algorithms...")
        
        # Create comparison view
        comparison = map_builder.visualize(start=start, goal=goal)
        
        # Draw all paths in different colors
        colors = {
            'A*': (0, 255, 0),      # Green
            'RRT*': (255, 0, 0),    # Blue
            'Greedy': (0, 255, 255) # Yellow
        }
        
        scale = 5
        y_offset = 30
        for name, path in results.items():
            # Draw path
            for i in range(len(path) - 1):
                pt1 = (int(path[i][0] * scale), int(path[i][1] * scale))
                pt2 = (int(path[i+1][0] * scale), int(path[i+1][1] * scale))
                cv2.line(comparison, pt1, pt2, colors[name], 1)
            
            # Add legend
            cv2.putText(comparison, f"{name}: {len(path)} waypoints", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       colors[name], 2)
            y_offset += 30
        
        cv2.imshow("Step 6: Algorithm Comparison", comparison)
        
        if not wait_for_key():
            return
    
    # Step 7: Simulate navigation
    print("\n[Step 7/7] Simulating navigation...")
    
    if 'A*' in results:
        path = results['A*']
        print(f"  Following A* path with {len(path)} waypoints")
        
        for i, waypoint in enumerate(path):
            # Show current position
            nav_vis = map_builder.visualize(path=path, start=waypoint, goal=goal)
            cv2.putText(nav_vis, f"Waypoint {i+1}/{len(path)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(nav_vis, f"Position: {waypoint}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add progress bar
            progress = int((i / len(path)) * 400)
            cv2.rectangle(nav_vis, (50, 100), (450, 120), (100, 100, 100), 2)
            cv2.rectangle(nav_vis, (50, 100), (50 + progress, 120), (0, 255, 0), -1)
            
            cv2.imshow("Step 7: Navigation Simulation", nav_vis)
            
            # Check for quit
            key = cv2.waitKey(50)
            if key == ord('q'):
                break
            elif key == ord('p'):  # Pause
                wait_for_key("Paused. Press any key to continue...")
        
        print("✓ Navigation simulation complete!")
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 VISUAL TEST COMPLETE!")
    print("="*70)
    print("\nSummary:")
    print(f"  Safe terrain: {stats['safe_percentage']:.1f}%")
    print(f"  Grid size: {map_stats['width']}x{map_stats['height']}")
    
    if results:
        print(f"  Algorithms tested: {len(results)}")
        for name, path in results.items():
            print(f"    {name}: {len(path)} waypoints")
    
    print("\nAll visualization windows still open.")
    print("Press any key to close all windows and exit...")
    cv2.waitKey(0)
    
    # Cleanup
    drone.disconnect()
    cv2.destroyAllWindows()
    
    print("\n✓ Test complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()