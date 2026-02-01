"""
Dynamic Real-Time Navigation System
Continuously replans, sends commands one at a time, and visualizes progress
"""

import sys
import time
import cv2
import numpy as np
import os
import subprocess

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
from dynamic_image_input import DynamicImageInput


class DynamicNavigationSystem:
    """
    Real-time dynamic navigation with continuous replanning.
    
    This system:
    1. Plans initial path to goal
    2. Sends FIRST command to robot
    3. Updates image/map
    4. Replans path from current position
    5. Sends NEXT command
    6. Repeat until goal reached
    """
    
    def __init__(self, pi_hostname="192.168.1.10", pi_username="asanku"):
        """Initialize the dynamic navigation system."""
        print("\n" + "="*70)
        print("🚁 DYNAMIC REAL-TIME NAVIGATION SYSTEM")
        print("="*70)
        print("Initializing components...\n")
        
        # Components
        self.image_input = DynamicImageInput("data/test_images/current_scene.jpg")
        self.segmenter = GeoSAMSegmenter()
        self.map_builder = MapBuilder()
        self.path_converter = PathConverter(self.map_builder, unit_scale=1.0)
        self.logger = DataLogger()
        
        # Planners
        self.planners = {
            'astar': AStarPlanner(self.map_builder),
            'rrt_star': RRTStarPlanner(self.map_builder),
            'greedy': GreedyPlanner(self.map_builder)
        }
        self.current_algorithm = config.DEFAULT_ALGORITHM
        
        # Robot communication
        self.pi_hostname = pi_hostname
        self.pi_username = pi_username
        self.pi_commands_dir = "/home/asanku/Documents/RSEF/movements"
        
        # Navigation state
        self.current_position_grid = None  # Current position in grid coords
        self.goal_position = None
        self.current_path = None
        self.current_heading = 0  # Current robot heading in degrees
        self.commands_sent = []
        self.iteration = 0
        
        # Visualization windows
        self.vis_width = 500
        self.vis_height = 500
        
        print("✓ All components initialized\n")
    
    def setup(self):
        """Setup the system."""
        print("Setting up system...")
        
        # Load image
        print("\n[1/3] Loading test image...")
        if not self.image_input.connect():
            return False
        self.image_input.start_stream()
        time.sleep(0.5)
        
        # Load model
        print("\n[2/3] Loading segmentation model...")
        if not self.segmenter.load_model():
            return False
        
        # Verify frame
        print("\n[3/3] Verifying image input...")
        test_frame = self.image_input.get_frame()
        if test_frame is None:
            print("✗ Cannot receive image")
            return False
        
        print("\n✓ System setup complete!\n")
        return True
    
    def set_navigation_goal(self, start_grid, goal_grid):
        """Set start and goal positions."""
        self.current_position_grid = start_grid
        self.goal_position = goal_grid
        self.current_heading = 0
        
        print(f"Navigation goal set:")
        print(f"  Start: {start_grid}")
        print(f"  Goal: {goal_grid}")
    
    def select_algorithm(self, algorithm_name):
        """Select planning algorithm."""
        if algorithm_name in self.planners:
            self.current_algorithm = algorithm_name
            print(f"✓ Algorithm: {algorithm_name}")
        else:
            print(f"⚠ Unknown algorithm, using {config.DEFAULT_ALGORITHM}")
            self.current_algorithm = config.DEFAULT_ALGORITHM
    
    def run_dynamic_navigation(self):
        """
        Main dynamic navigation loop.
        Plans → Send one command → Update → Replan → Repeat
        """
        print("\n" + "="*70)
        print("🤖 STARTING DYNAMIC NAVIGATION")
        print("="*70)
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  Q - Quit")
        print("  R - Force replan")
        print("\n")
        
        self.iteration = 0
        paused = False
        
        # Create visualization windows
        cv2.namedWindow("1. Current Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("2. Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("3. Navigation Map", cv2.WINDOW_NORMAL)
        cv2.namedWindow("4. Progress Tracker", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                self.iteration += 1
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠ User requested quit")
                    break
                elif key == ord(' '):
                    paused = not paused
                    if paused:
                        print("\n⏸ PAUSED - Press SPACE to resume")
                    else:
                        print("▶ RESUMED")
                elif key == ord('r'):
                    print("\n🔄 Forced replan requested")
                
                if paused:
                    time.sleep(0.1)
                    continue
                
                print(f"\n{'='*70}")
                print(f"ITERATION {self.iteration}")
                print(f"{'='*70}")
                print(f"Current position: {self.current_position_grid}")
                print(f"Current heading: {self.current_heading}°")
                
                # Step 1: Get fresh image and segment
                print("\n[Step 1/5] Capturing and segmenting image...")
                if not self._update_map():
                    print("⚠ Map update failed, using previous map")
                
                # Step 2: Check if we reached goal
                if self._check_goal_reached():
                    print("\n" + "="*70)
                    print("🎉 GOAL REACHED!")
                    print("="*70)
                    self._show_success_visualization()
                    break
                
                # Step 3: Plan path from current position to goal
                print(f"\n[Step 2/5] Planning path using {self.current_algorithm}...")
                path = self._plan_path()
                
                if not path or len(path) < 2:
                    print("✗ No path found! Navigation failed.")
                    break
                
                self.current_path = path
                print(f"✓ Path planned: {len(path)} waypoints")
                
                # Step 4: Generate commands for entire path
                print("\n[Step 3/5] Converting path to commands...")
                commands = self.path_converter.convert_path_to_commands(
                    path, 
                    start_heading=self.current_heading
                )
                
                if not commands:
                    print("✗ No commands generated!")
                    break
                
                print(f"✓ Generated {len(commands)} commands")
                
                # Step 5: Send ONLY the first command to robot
                print("\n[Step 4/5] Sending first command to robot...")
                first_command = commands[0]
                
                success = self._send_single_command(first_command)
                
                if not success:
                    print("✗ Failed to send command to robot")
                    break
                
                self.commands_sent.append(first_command)
                print(f"✓ Sent: {first_command}")
                
                # Step 6: Update robot state based on command
                print("\n[Step 5/5] Updating robot state...")
                self._update_robot_state(first_command, path)
                
                # Step 7: Visualize current state
                self._visualize_state(path)
                
                # Wait a moment before next iteration
                print(f"\n⏳ Waiting before next iteration...")
                time.sleep(2)  # Give robot time to execute
        
        except KeyboardInterrupt:
            print("\n\n⚠ Navigation interrupted by user")
        
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("\n" + "="*70)
            print("Navigation Summary:")
            print(f"  Iterations: {self.iteration}")
            print(f"  Commands sent: {len(self.commands_sent)}")
            print(f"  Final position: {self.current_position_grid}")
            print("="*70)
            
            # Keep windows open
            print("\nPress any key to close visualization windows...")
            cv2.waitKey(0)
    
    def _update_map(self):
        """Capture new image and update map."""
        # Get image
        image = self.image_input.get_frame()
        if image is None:
            return False
        
        # Segment
        mask = self.segmenter.segment(image)
        if mask is None:
            return False
        
        stats = self.segmenter.get_statistics(mask)
        print(f"  Segmentation: {stats['safe_percentage']:.1f}% safe")
        
        # Update map
        success = self.map_builder.update_from_segmentation(mask)
        
        # Store for visualization
        self.current_image = image
        self.current_mask = mask
        
        return success
    
    def _check_goal_reached(self):
        """Check if robot reached the goal."""
        if self.current_position_grid is None or self.goal_position is None:
            return False
        
        dx = abs(self.current_position_grid[0] - self.goal_position[0])
        dy = abs(self.current_position_grid[1] - self.goal_position[1])
        distance = np.sqrt(dx**2 + dy**2)
        
        # Consider goal reached if within 2 cells
        return distance < 2
    
    def _plan_path(self):
        """Plan path from current position to goal."""
        planner = self.planners[self.current_algorithm]
        
        plan_start = time.time()
        path = planner.plan(self.current_position_grid, self.goal_position)
        plan_time = time.time() - plan_start
        
        if path:
            self.logger.log_planning(
                self.current_algorithm,
                plan_time,
                len(path),
                True
            )
        else:
            self.logger.log_planning(
                self.current_algorithm,
                plan_time,
                0,
                False
            )
        
        return path
    
    def _send_single_command(self, command):
        """
        Send a single command to the Raspberry Pi.
        Overwrites the robot_path.txt file with just this one command.
        """
        try:
            # Create temporary file with single command
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            temp_file.write(command + '\n')
            temp_file.close()
            
            # Send via SCP
            remote_path = f"{self.pi_username}@{self.pi_hostname}:{self.pi_commands_dir}/robot_path.txt"
            
            result = subprocess.run(
                ["scp", temp_file.name, remote_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Clean up
            os.unlink(temp_file.name)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"  ✗ Send error: {str(e)}")
            return False
    
    def _update_robot_state(self, command, path):
        """
        Update robot's position and heading based on executed command.
        """
        if command.startswith('turn('):
            # Extract angle
            angle = int(command[5:-1])
            self.current_heading = angle
            print(f"  Robot turned to {angle}°")
        
        elif command.startswith('move('):
            # Robot moved forward - update position
            # Move to next waypoint in path
            if len(path) > 1:
                self.current_position_grid = path[1]
                print(f"  Robot moved to {self.current_position_grid}")
            else:
                print(f"  Robot at final waypoint")
    
    def _visualize_state(self, path):
        """
        Create comprehensive visualization of current state.
        """
        # Window 1: Current image
        if hasattr(self, 'current_image'):
            img_vis = cv2.resize(self.current_image, (self.vis_width, self.vis_height))
            cv2.putText(img_vis, f"Iteration {self.iteration}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("1. Current Image", img_vis)
        
        # Window 2: Segmentation
        if hasattr(self, 'current_mask'):
            seg_vis = self.segmenter.visualize_segmentation(
                self.current_image, self.current_mask
            )
            seg_vis = cv2.resize(seg_vis, (self.vis_width, self.vis_height))
            cv2.imshow("2. Segmentation", seg_vis)
        
        # Window 3: Navigation map with current path
        map_vis = self.map_builder.visualize(
            path=path,
            start=self.current_position_grid,
            goal=self.goal_position
        )
        
        # Add info
        cv2.putText(map_vis, f"Algorithm: {self.current_algorithm}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(map_vis, f"Position: {self.current_position_grid}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(map_vis, f"Heading: {self.current_heading}deg", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(map_vis, f"Waypoints: {len(path)}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("3. Navigation Map", map_vis)
        
        # Window 4: Progress tracker
        progress_vis = self._create_progress_visualization(path)
        cv2.imshow("4. Progress Tracker", progress_vis)
    
    def _create_progress_visualization(self, path):
        """
        Create a progress tracking visualization.
        Shows: commands sent, distance to goal, progress bar
        """
        vis = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Title
        cv2.putText(vis, "Navigation Progress", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # Iteration count
        cv2.putText(vis, f"Iteration: {self.iteration}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Commands sent
        cv2.putText(vis, f"Commands sent: {len(self.commands_sent)}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Last command
        if self.commands_sent:
            cv2.putText(vis, f"Last: {self.commands_sent[-1]}", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
        
        # Distance to goal
        if self.current_position_grid and self.goal_position:
            dx = self.goal_position[0] - self.current_position_grid[0]
            dy = self.goal_position[1] - self.current_position_grid[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            cv2.putText(vis, f"Distance to goal: {distance:.1f} cells", (20, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Progress bar
            start_dist = np.sqrt(
                (self.goal_position[0] - config.TEST_START_POSITION[0])**2 +
                (self.goal_position[1] - config.TEST_START_POSITION[1])**2
            )
            
            if start_dist > 0:
                progress = 1 - (distance / start_dist)
                progress = max(0, min(1, progress))
                
                # Draw progress bar
                bar_x = 20
                bar_y = 220
                bar_w = 560
                bar_h = 40
                
                # Background
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                            (200, 200, 200), -1)
                
                # Progress
                progress_w = int(bar_w * progress)
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h),
                            (0, 255, 0), -1)
                
                # Border
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                            (0, 0, 0), 2)
                
                # Percentage
                cv2.putText(vis, f"{progress*100:.1f}%", (bar_x + bar_w//2 - 40, bar_y + 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Recent commands list
        cv2.putText(vis, "Recent Commands:", (20, 290),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        recent = self.commands_sent[-5:] if len(self.commands_sent) > 5 else self.commands_sent
        for i, cmd in enumerate(recent):
            cv2.putText(vis, f"  {len(self.commands_sent) - len(recent) + i + 1}. {cmd}",
                       (30, 320 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return vis
    
    def _show_success_visualization(self):
        """Show success message on visualization."""
        success_vis = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        cv2.putText(success_vis, "GOAL REACHED!", (100, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 3)
        
        cv2.putText(success_vis, f"Total iterations: {self.iteration}", (150, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        cv2.putText(success_vis, f"Commands sent: {len(self.commands_sent)}", (150, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        cv2.imshow("4. Progress Tracker", success_vis)
    
    def shutdown(self):
        """Cleanup."""
        print("\nShutting down...")
        self.logger.save_to_csv()
        self.logger.save_events()
        self.logger.print_summary()
        self.image_input.disconnect()
        cv2.destroyAllWindows()
        print("✓ Shutdown complete")


def main():
    """Main entry point."""
    system = DynamicNavigationSystem(
        pi_hostname="192.168.1.10",
        pi_username="asanku"
    )
    
    try:
        # Setup
        if not system.setup():
            print("✗ Setup failed")
            return 1
        
        # Select algorithm
        algorithm = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_ALGORITHM
        system.select_algorithm(algorithm)
        
        # Set goal
        system.set_navigation_goal(
            config.TEST_START_POSITION,
            config.TEST_GOAL_POSITION
        )
        
        # Run dynamic navigation
        system.run_dynamic_navigation()
        
        return 0
    
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        system.shutdown()


if __name__ == "__main__":
    config.create_directories()
    exit_code = main()
    sys.exit(exit_code)
