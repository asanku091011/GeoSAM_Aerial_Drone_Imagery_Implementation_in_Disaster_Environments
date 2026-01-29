"""
Main integration file for the Disaster Navigation System.
Connects all components and runs the complete pipeline.

SIMPLIFIED VERSION - Uses the proven approach from debug_path.py
"""

import sys
import time
import cv2
import numpy as np
import os

# Suppress OpenCV/FFmpeg warnings for cleaner output
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import config

# Import all system components
from image_input import ImageInput
from segmentation import GeoSAMSegmenter
from map_builder import MapBuilder
from astar import AStarPlanner
from rrt_star import RRTStarPlanner
from greedy import GreedyPlanner
from path_converter import PathConverter
from data_logger import DataLogger


class DisasterNavigationSystem:
    """
    Main system class that integrates all components.
    
    SIMPLIFIED: Uses single image segmentation like debug_path.py
    """
    
    def __init__(self):
        """Initialize all system components."""
        print("\n" + "="*70)
        print("🚁 DISASTER NAVIGATION SYSTEM")
        print("="*70)
        print("Initializing components...\n")
        
        # Create all components
        self.image_input = ImageInput()
        self.segmenter = GeoSAMSegmenter()
        self.map_builder = MapBuilder()
        self.path_converter = PathConverter(self.map_builder, unit_scale=1.0)
        self.logger = DataLogger()
        
        # Create planners for all algorithms
        self.planners = {
            'astar': AStarPlanner(self.map_builder),
            'rrt_star': RRTStarPlanner(self.map_builder),
            'greedy': GreedyPlanner(self.map_builder)
        }
        
        # System state
        self.current_algorithm = config.DEFAULT_ALGORITHM
        self.current_path = None
        self.goal_position = None
        
        print("✓ All components initialized\n")
    
    def setup(self):
        """
        Set up the system before starting navigation.
        
        Returns:
            bool: True if setup successful
        """
        print("Setting up system...")
        
        # 1. Load image
        print("\n[1/3] Loading test image...")
        if not self.image_input.connect():
            print("✗ Image loading failed")
            return False
        
        self.image_input.start_stream()
        time.sleep(0.5)
        
        # 2. Load segmentation model
        print("\n[2/3] Loading segmentation model...")
        if not self.segmenter.load_model():
            print("✗ Model loading failed")
            return False
        
        # 3. Get initial frame
        print("\n[3/3] Getting initial frame...")
        test_frame = self.image_input.get_frame()
        if test_frame is None:
            print("✗ Cannot receive image")
            return False
        
        print("\n✓ System setup complete!\n")
        return True
    
    def set_navigation_goal(self, start_grid, goal_grid):
        """
        Set the start and goal positions for navigation.
        
        Args:
            start_grid (tuple): Start position in grid coordinates (x, y)
            goal_grid (tuple): Goal position in grid coordinates (x, y)
        """
        self.start_position = start_grid
        self.goal_position = goal_grid
        
        print(f"Navigation goal set:")
        print(f"  Start: {start_grid} (grid)")
        print(f"  Goal: {goal_grid} (grid)")
    
    def select_algorithm(self, algorithm_name):
        """
        Select which path planning algorithm to use.
        
        Args:
            algorithm_name (str): 'astar', 'rrt_star', or 'greedy'
        """
        if algorithm_name not in self.planners:
            print(f"⚠ Unknown algorithm '{algorithm_name}', using {config.DEFAULT_ALGORITHM}")
            algorithm_name = config.DEFAULT_ALGORITHM
        
        self.current_algorithm = algorithm_name
        print(f"✓ Algorithm selected: {algorithm_name}")
    
    def run_navigation(self):
        """
        Run the complete navigation pipeline using debug_path.py's approach.
        
        Returns:
            bool: True if navigation completed successfully
        """
        print("\n" + "="*70)
        print("🤖 RUNNING NAVIGATION")
        print("="*70)
        
        # Step 1: Get image
        print("\n[Step 1/5] Getting image...")
        image = self.image_input.get_frame()
        if image is None:
            print("✗ Failed to get image")
            return False
        print(f"✓ Image loaded: {image.shape}")
        
        # Step 2: Segment image
        print("\n[Step 2/5] Segmenting image...")
        mask = self.segmenter.segment(image)
        if mask is None:
            print("✗ Segmentation failed")
            return False
        
        stats = self.segmenter.get_statistics(mask)
        print(f"✓ Segmentation complete")
        print(f"  Safe terrain: {stats['safe_percentage']:.1f}%")
        print(f"  Unsafe terrain: {stats['unsafe_percentage']:.1f}%")
        
        # Step 3: Build navigation map
        print("\n[Step 3/5] Building navigation map...")
        success = self.map_builder.update_from_segmentation(mask)
        if not success:
            print("✗ Map building failed")
            return False
        
        map_stats = self.map_builder.get_statistics()
        print(f"✓ Map built: {map_stats['width']}x{map_stats['height']} grid")
        print(f"  Free cells: {map_stats['free_percentage']:.1f}%")
        
        # Step 4: Validate start and goal positions
        print("\n[Step 4/5] Validating positions...")
        print(f"  Start: {self.start_position}")
        print(f"  Goal: {self.goal_position}")
        
        start_free = self.map_builder.is_cell_free(self.start_position[0], self.start_position[1])
        goal_free = self.map_builder.is_cell_free(self.goal_position[0], self.goal_position[1])
        
        print(f"  Start is free: {start_free}")
        print(f"  Goal is free: {goal_free}")
        
        if not start_free or not goal_free:
            print("✗ Start or goal position is blocked!")
            return False
        
        # Step 5: Plan path
        print(f"\n[Step 5/5] Planning path with {self.current_algorithm}...")
        planner = self.planners[self.current_algorithm]
        
        plan_start = time.time()
        path = planner.plan(self.start_position, self.goal_position)
        plan_time = time.time() - plan_start
        
        if not path:
            print("✗ No path found!")
            self.logger.log_planning(self.current_algorithm, plan_time, 0, False)
            return False
        
        print(f"✓ Path found: {len(path)} waypoints")
        print(f"  Planning time: {plan_time*1000:.1f}ms")
        
        # Log planning
        self.logger.log_planning(self.current_algorithm, plan_time, len(path), True)
        
        # Analyze path
        self._analyze_path(path)
        
        # Save path
        self.current_path = path
        
        # Convert to robot commands
        print("\n[Bonus] Converting to robot commands...")
        commands = self.path_converter.convert_path_to_commands(path, start_heading=0)
        commands = self.path_converter.optimize_commands(commands)
        self.path_converter.preview_commands(commands, max_lines=10)
        self.path_converter.save_commands(commands)
        
        # Visualize
        if config.ENABLE_VISUALIZATION:
            self._visualize_results(image, mask, path)
        
        return True
    
    def _analyze_path(self, path):
        """
        Analyze the planned path (like debug_path.py does).
        
        Args:
            path (list): Planned path
        """
        print(f"\n  Path analysis:")
        print(f"    First 5 waypoints: {path[:5]}")
        print(f"    Last 5 waypoints: {path[-5:]}")
        
        # Calculate path length
        total_distance = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        print(f"    Total distance: {total_distance:.1f} cells")
        
        # Straight line distance
        straight_dist = np.sqrt(
            (self.goal_position[0] - self.start_position[0])**2 + 
            (self.goal_position[1] - self.start_position[1])**2
        )
        print(f"    Straight line: {straight_dist:.1f} cells")
        print(f"    Detour factor: {total_distance / straight_dist:.2f}x")
        
        # Check if path goes through obstacles
        obstacles_hit = 0
        for waypoint in path:
            if not self.map_builder.is_cell_free(waypoint[0], waypoint[1]):
                obstacles_hit += 1
        
        if obstacles_hit > 0:
            print(f"    ⚠ WARNING: Path goes through {obstacles_hit} obstacle cells!")
        else:
            print(f"    ✓ Path avoids all obstacles")
    
    def _visualize_results(self, image, mask, path):
        """
        Create visualizations of the results.
        
        Args:
            image: Original image
            mask: Segmentation mask
            path: Planned path
        """
        print("\n[Visualization] Creating windows...")
        
        # 1. Original image
        cv2.imshow("1. Original Image", image)
        
        # 2. Segmentation overlay
        seg_vis = self.segmenter.visualize_segmentation(image, mask)
        cv2.imshow("2. Segmentation Overlay", seg_vis)
        
        # 3. Binary mask
        mask_vis = mask * 255
        mask_vis_color = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        cv2.imshow("3. Binary Mask", mask_vis_color)
        
        # 4. Planned path on map
        path_vis = self.map_builder.visualize(
            path=path,
            start=self.start_position,
            goal=self.goal_position
        )
        
        # Add statistics to path visualization
        cv2.putText(path_vis, f"Algorithm: {self.current_algorithm}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(path_vis, f"Waypoints: {len(path)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(path_vis, f"Press 'Q' to quit, 'S' to save", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("4. Planned Path", path_vis)
        
        print("\n" + "="*70)
        print("✓ Visualizations ready!")
        print("  Press 'Q' to quit")
        print("  Press 'S' to save visualization")
        print("="*70)
        
        # Wait for user input
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save visualization
                timestamp = int(time.time())
                filename = f"outputs/path_visualization_{timestamp}.jpg"
                cv2.imwrite(filename, path_vis)
                print(f"\n✓ Saved visualization to {filename}")
    
    def shutdown(self):
        """Clean shutdown of all system components."""
        print("\n" + "="*70)
        print("Shutting down system...")
        print("="*70)
        
        # Save data
        print("\nSaving data...")
        self.logger.save_to_csv()
        self.logger.save_events()
        
        # Print summary
        self.logger.print_summary()
        
        # Cleanup
        print("\nCleaning up...")
        self.image_input.disconnect()
        cv2.destroyAllWindows()
        
        print("\n✓ System shutdown complete")


def main():
    """
    Main entry point for the Disaster Navigation System.
    Simplified version using debug_path.py's approach.
    """
    # Create system
    system = DisasterNavigationSystem()
    
    try:
        # Setup system
        if not system.setup():
            print("✗ System setup failed")
            return 1
        
        # Select algorithm (can be changed via command line argument)
        algorithm = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_ALGORITHM
        system.select_algorithm(algorithm)
        
        # Set navigation goal
        start = config.TEST_START_POSITION
        goal = config.TEST_GOAL_POSITION
        system.set_navigation_goal(start, goal)
        
        # Run navigation
        success = system.run_navigation()
        
        if success:
            print("\n✓ Navigation completed successfully!")
            return 0
        else:
            print("\n✗ Navigation failed")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Always shutdown cleanly
        system.shutdown()


if __name__ == "__main__":
    # Create required directories
    config.create_directories()
    
    # Run main system
    exit_code = main()
    sys.exit(exit_code)