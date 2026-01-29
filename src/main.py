"""
Main integration file for the Disaster Navigation System.
Connects all components and runs the complete pipeline.

This is the entry point for the system. It:
1. Initializes the drone and connects to video stream
2. Continuously captures and segments overhead images
3. Builds and updates navigation maps
4. Plans and executes robot paths
5. Handles dynamic replanning when obstacles appear
6. Logs all data for analysis
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
from drone_input import DroneInput
from image_input import ImageInput  # NEW: For still images
from segmentation import GeoSAMSegmenter
from map_builder import MapBuilder
from astar import AStarPlanner
from rrt_star import RRTStarPlanner
from greedy import GreedyPlanner
from dynamic_replanner import DynamicReplanner
from path_converter import PathConverter  # NEW: For robot_path.txt
from data_logger import DataLogger


class DisasterNavigationSystem:
    """
    Main system class that integrates all components.
    
    This is the brain of the operation - it coordinates between:
    - Drone video input
    - AI segmentation
    - Map building
    - Path planning
    - Robot control
    - Data logging
    """
    
    def __init__(self):
        """Initialize all system components."""
        print("\n" + "="*70)
        print("🚁 DISASTER NAVIGATION SYSTEM")
        print("="*70)
        print("Initializing components...\n")
        
        # Create all components
        # Choose input source based on config
        if config.DRONE_ENABLED:
            self.drone = DroneInput(use_drone=True)
            print("Using: Real drone video")
        elif config.TEST_MODE:
            self.drone = DroneInput(use_drone=False)  # Synthetic
            print("Using: Synthetic video")
        else:
            self.drone = ImageInput()  # NEW: Still images
            print("Using: Still images from data/test_images/")
        
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
        
        # Dynamic replanner (will be set after choosing planner)
        self.replanner = None
        
        # System state
        self.is_running = False
        self.current_algorithm = config.DEFAULT_ALGORITHM
        self.current_path = None
        self.current_position = None
        self.goal_position = None
        
        print("✓ All components initialized\n")
    
    def setup(self):
        """
        Set up the system before starting navigation.
        
        Returns:
            bool: True if setup successful
        """
        print("Setting up system...")
        
        # 1. Connect to drone
        print("\n[1/3] Connecting to drone...")
        if not self.drone.connect():
            print("✗ Drone connection failed")
            return False
        
        # 2. Load segmentation model
        print("\n[2/3] Loading segmentation model...")
        if not self.segmenter.load_model():
            print("✗ Model loading failed")
            return False
        
        # 3. Start video stream
        print("\n[3/3] Starting video stream...")
        self.drone.start_stream()
        time.sleep(2)  # Wait for stream to stabilize
        
        # Verify we can get frames
        test_frame = self.drone.get_frame()
        if test_frame is None:
            print("✗ Cannot receive video frames")
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
        self.current_position = self.map_builder.grid_to_world(
            start_grid[0], start_grid[1]
        )
        self.goal_position = goal_grid
        
        print(f"Navigation goal set:")
        print(f"  Start: {start_grid} (grid) -> {self.current_position} (world)")
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
        
        # Create dynamic replanner with selected planner
        self.replanner = DynamicReplanner(
            self.map_builder, 
            self.planners[algorithm_name]
        )
        
        print(f"✓ Algorithm selected: {algorithm_name}")
    
    def update_map(self):
        """
        Capture a new frame, segment it, and update the navigation map.
        
        Returns:
            bool: True if map update successful
        """
        # Get latest frame from drone
        frame = self.drone.get_frame()
        if frame is None:
            return False
        
        # Segment the frame to identify safe/unsafe areas
        seg_start = time.time()
        mask = self.segmenter.segment(frame)
        seg_time = time.time() - seg_start
        
        if mask is None:
            return False
        
        # VISUALIZATION: Show segmentation result
        if config.ENABLE_VISUALIZATION:
            seg_vis = self.segmenter.visualize_segmentation(frame, mask)
            cv2.imshow("1. Drone Feed + Segmentation", seg_vis)
            cv2.waitKey(1)
        
        # Update the navigation map
        success = self.map_builder.update_from_segmentation(mask)
        
        # Log segmentation performance
        if success:
            stats = self.segmenter.get_statistics(mask)
            self.logger.log_segmentation(seg_time, stats['safe_percentage'])
            print(f"  Segmentation: {stats['safe_percentage']:.1f}% safe terrain")
        
        return success
    
    def plan_initial_path(self):
        """
        Plan the initial path from start to goal.
        
        Returns:
            list: Planned path, or None if planning failed
        """
        print(f"\nPlanning initial path using {self.current_algorithm}...")
        
        # Get current position in grid coordinates
        start_grid = self.map_builder.world_to_grid(
            self.current_position[0], 
            self.current_position[1]
        )
        
        # VISUALIZATION: Show the map before planning
        if config.ENABLE_VISUALIZATION:
            pre_plan_vis = self.map_builder.visualize(
                start=start_grid,
                goal=self.goal_position
            )
            cv2.putText(pre_plan_vis, "Planning path...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("2. Navigation Map & Path", pre_plan_vis)
            cv2.waitKey(500)  # Show for half a second
        
        # Plan path
        planner = self.planners[self.current_algorithm]
        plan_start = time.time()
        path = planner.plan(start_grid, self.goal_position)
        plan_time = time.time() - plan_start
        
        # Log planning performance
        if path:
            self.logger.log_planning(
                self.current_algorithm, 
                plan_time, 
                len(path), 
                True
            )
            print(f"✓ Path found: {len(path)} waypoints in {plan_time:.3f}s")
            
            # VISUALIZATION: Show the planned path
            if config.ENABLE_VISUALIZATION:
                path_vis = self.map_builder.visualize(
                    path=path,
                    start=start_grid,
                    goal=self.goal_position
                )
                cv2.putText(path_vis, f"Path: {len(path)} waypoints", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(path_vis, f"Time: {plan_time*1000:.0f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(path_vis, "Press SPACE to start navigation...", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("2. Navigation Map & Path", path_vis)
                print("\nPath found! Press SPACE to start navigation, 'Q' to quit...")
                
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord(' '):  # Spacebar to continue
                        break
                    elif key == ord('q'):
                        print("User cancelled navigation")
                        return None
            
            # Set path for replanner
            self.current_path = path
            self.replanner.set_path(path, start_grid, self.goal_position)
            
            # Convert path to robot commands
            print("\nConverting path to robot commands...")
            commands = self.path_converter.convert_path_to_commands(path, start_heading=0)
            commands = self.path_converter.optimize_commands(commands)
            self.path_converter.preview_commands(commands, max_lines=10)
            self.path_converter.save_commands(commands)
        else:
            self.logger.log_planning(
                self.current_algorithm, 
                plan_time, 
                0, 
                False
            )
            print(f"✗ Planning failed after {plan_time:.3f}s")
            
            # VISUALIZATION: Show failed planning attempt
            if config.ENABLE_VISUALIZATION:
                fail_vis = self.map_builder.visualize(
                    start=start_grid,
                    goal=self.goal_position
                )
                cv2.putText(fail_vis, "PLANNING FAILED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(fail_vis, "Check start/goal positions", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("2. Navigation Map & Path", fail_vis)
                cv2.waitKey(2000)
        
        return path
    
    def navigation_loop(self):
        """
        Main navigation loop that runs continuously.
        This is where the magic happens!
        """
        print("\n" + "="*70)
        print("🤖 STARTING NAVIGATION")
        print("="*70)
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  Q - Quit")
        print("  S - Save current view")
        print("\n")
        
        self.is_running = True
        iteration = 0
        nav_start_time = time.time()
        paused = False
        
        # Log navigation start
        start_grid = self.map_builder.world_to_grid(
            self.current_position[0], 
            self.current_position[1]
        )
        self.logger.log_navigation_start(
            start_grid, 
            self.goal_position, 
            self.current_algorithm
        )
        
        try:
            while self.is_running:
                iteration += 1
                loop_start = time.time()
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠ User requested quit")
                    break
                elif key == ord(' '):
                    paused = not paused
                    if paused:
                        print("\n⏸ PAUSED - Press SPACE to resume...")
                    else:
                        print("▶ RESUMED")
                
                if paused:
                    time.sleep(0.1)
                    continue
                
                print(f"\n--- Iteration {iteration} ---")
                
                # Step 1: Update map from drone video
                if not self.update_map():
                    print("⚠ Map update failed")
                    time.sleep(0.1)
                    continue
                
                # Step 2: Check if replanning needed (if enabled)
                if config.DYNAMIC_REPLANNING_ENABLED and self.replanner:
                    needs_replan, new_path, reason = self.replanner.check_and_replan()
                    
                    if needs_replan:
                        if new_path:
                            old_length = len(self.current_path) if self.current_path else 0
                            self.current_path = new_path
                            self.controller.set_path(new_path)
                            
                            # Log replanning
                            self.logger.log_replanning(reason, old_length, len(new_path))
                        else:
                            # Replanning failed - this is critical!
                            self.logger.log_error(
                                "replanning_failed", 
                                "Cannot find alternative path",
                                {'reason': reason}
                            )
                            print("✗ CRITICAL: Replanning failed, stopping")
                            break
                
                # Step 3: Visualize the current state
                if config.ENABLE_VISUALIZATION:
                    self.visualize_system()
                
                # Step 4: Simulate robot progress along path
                # (In real deployment, this would come from robot odometry)
                if self.current_path and len(self.current_path) > 1:
                    # Move to next waypoint
                    waypoint_idx = min(iteration, len(self.current_path) - 1)
                    waypoint = self.current_path[waypoint_idx]
                    self.current_position = self.map_builder.grid_to_world(waypoint[0], waypoint[1])
                    
                    # Check if reached goal
                    if waypoint == self.goal_position:
                        print("\n✓ Goal reached!")
                        break
                elapsed_time = time.time() - nav_start_time
                if elapsed_time > config.MAX_EXECUTION_TIME:
                    print(f"⚠ Max execution time ({config.MAX_EXECUTION_TIME}s) reached")
                    break
                
                # Control loop timing
                loop_time = time.time() - loop_start
                sleep_time = max(0, 1.0/config.SYSTEM_UPDATE_RATE - loop_time)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n⚠ Navigation interrupted by user")
        
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            self.logger.log_error("system_error", str(e))
        
        finally:
            # Calculate final metrics
            total_time = time.time() - nav_start_time
            path_length = self._calculate_path_length(self.current_path) if self.current_path else 0
            
            # Log navigation end
            goal_reached = self._check_goal_reached()
            self.logger.log_navigation_end(goal_reached, path_length, total_time)
            
            self.is_running = False
    
    def visualize_system(self):
        """
        Create and display visualization of the current state.
        Shows both drone feed and navigation map.
        """
        # Get latest drone frame
        drone_frame = self.drone.get_frame()
        
        # Show drone feed with segmentation overlay if available
        if drone_frame is not None and config.ENABLE_VISUALIZATION:
            # Create a quick segmentation for visualization
            try:
                mask = self.segmenter.segment(drone_frame)
                if mask is not None:
                    overlay = self.segmenter.visualize_segmentation(drone_frame, mask)
                    fps = self.drone.get_fps()
                    cv2.putText(overlay, f"Drone FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("1. Drone Feed + Segmentation", overlay)
            except:
                # If segmentation fails, just show raw feed
                cv2.imshow("1. Drone Feed", drone_frame)
        
        # Get current position in grid coordinates
        current_grid = self.map_builder.world_to_grid(
            self.current_position[0],
            self.current_position[1]
        )
        
        # Visualize map with path
        vis = self.map_builder.visualize(
            path=self.current_path,
            start=current_grid,
            goal=self.goal_position
        )
        
        # Add text overlay with stats
        fps = self.drone.get_fps()
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(vis, f"Algorithm: {self.current_algorithm}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.replanner:
            stats = self.replanner.get_statistics()
            cv2.putText(vis, f"Replans: {stats['replan_count']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.current_path:
            cv2.putText(vis, f"Waypoints: {len(self.current_path)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show position
        cv2.putText(vis, f"Pos: {current_grid}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add controls reminder
        cv2.putText(vis, "Q=Quit SPACE=Pause", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("2. Navigation Map & Path", vis)
        
        # Save visualization if enabled
        if config.SAVE_VISUALIZATION:
            timestamp = int(time.time() * 1000)
            filename = f"{config.VISUALIZATION_OUTPUT_DIR}/frame_{timestamp}.jpg"
            cv2.imwrite(filename, vis)
    
    def _calculate_path_length(self, path):
        """Calculate total path length in meters."""
        if not path or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            p1 = self.map_builder.grid_to_world(path[i][0], path[i][1])
            p2 = self.map_builder.grid_to_world(path[i+1][0], path[i+1][1])
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            total_length += np.sqrt(dx**2 + dy**2)
        
        return total_length
    
    def _check_goal_reached(self):
        """Check if robot reached the goal."""
        current_grid = self.map_builder.world_to_grid(
            self.current_position[0],
            self.current_position[1]
        )
        
        dx = abs(current_grid[0] - self.goal_position[0])
        dy = abs(current_grid[1] - self.goal_position[1])
        distance = np.sqrt(dx**2 + dy**2)
        
        return distance < config.WAYPOINT_REACHED_THRESHOLD / config.GRID_RESOLUTION
    
    def shutdown(self):
        """Clean shutdown of all system components."""
        print("\n" + "="*70)
        print("Shutting down system...")
        print("="*70)
        
        # Stop navigation
        self.is_running = False
        
        # Save all data
        print("\nSaving data...")
        if self.current_path:
            commands = self.path_converter.convert_path_to_commands(self.current_path)
            commands = self.path_converter.optimize_commands(commands)
            self.path_converter.save_commands(commands)
        
        self.logger.save_to_csv()
        self.logger.save_events()
        
        # Print summary
        self.logger.print_summary()
        
        # Cleanup hardware
        print("\nCleaning up hardware...")
        self.drone.disconnect()
        cv2.destroyAllWindows()
        
        print("\n✓ System shutdown complete")


def main():
    """
    Main entry point for the Disaster Navigation System.
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
        # In test mode, use config values
        # In real deployment, these would come from mission planning
        if config.TEST_MODE:
            start = config.TEST_START_POSITION
            goal = config.TEST_GOAL_POSITION
        else:
            # For real operation, you might determine start from drone position
            # and goal from user input or mission file
            start = (10, 10)
            goal = (90, 90)
        
        system.set_navigation_goal(start, goal)
        
        # Plan initial path
        path = system.plan_initial_path()
        if path is None:
            print("✗ Initial planning failed")
            return 1
        
        # Run navigation loop
        system.navigation_loop()
        
        return 0
    
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