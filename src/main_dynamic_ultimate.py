"""
ULTIMATE Dynamic Navigation System
- Coordinate system fixed ✓
- Dynamic image reloading ✓
- Smooth 360° paths ✓
- Real-time replanning ✓
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
from dynamic_image_input import DynamicImageInput  # ← DYNAMIC IMAGES!
from segmentation import GeoSAMSegmenter
from map_builder import MapBuilder
from astar_smooth import AStarSmoothPlanner  # ← SMOOTH 360° PATHS!
from rrt_star import RRTStarPlanner
from greedy import GreedyPlanner
from path_converter_smooth import SmoothPathConverter  # ← SMOOTH CONVERTER!
from data_logger import DataLogger


class DynamicNavigationSystem:
    """
    Ultimate real-time dynamic navigation with:
    - Continuous replanning
    - Dynamic image updates
    - Smooth 360° paths
    - Correct coordinate system
    """
    
    def __init__(self):
        print("\n" + "="*70)
        print("🚁 ULTIMATE DYNAMIC NAVIGATION SYSTEM")
        print("="*70)
        print("Features:")
        print("  ✓ Dynamic image reloading")
        print("  ✓ Smooth 360° path planning")
        print("  ✓ Fixed coordinate system")
        print("  ✓ Real-time replanning")
        print("="*70)
        print("\nInitializing components...\n")
        
        # DYNAMIC IMAGE INPUT
        self.image_input = DynamicImageInput("data/test_images/ladi_00297_segmented.jpg")
        
        # Segmentation
        self.segmenter = GeoSAMSegmenter()
        
        # Map builder
        self.map_builder = MapBuilder()
        
        # SMOOTH PATH CONVERTER (360° angles)
        self.path_converter = SmoothPathConverter(self.map_builder, unit_scale=1.0)
        
        # Data logger
        config.create_directories()
        self.logger = DataLogger()
        
        # Create planners - SMOOTH A*!
        self.planners = {
            'astar': AStarSmoothPlanner(self.map_builder),  # ← SMOOTH!
            'rrt_star': RRTStarPlanner(self.map_builder),
            'greedy': GreedyPlanner(self.map_builder)
        }
        
        # State tracking
        self.current_algorithm = config.DEFAULT_ALGORITHM
        self.current_position_grid = None
        self.current_heading = 0.0  # Continuous angles now!
        self.goal_position = None
        self.iteration = 0
        self.commands_sent = []
        self.robot_connected = False  # Will be checked during setup
        self.current_path = None  # Store path between iterations
        
        # Visualization
        self.vis_width = 640
        self.vis_height = 480
        
        print("✓ All components initialized\n")
    
    def setup(self):
        """Setup system."""
        print("Setting up system...")
        
        print("\n[1/4] Loading dynamic image...")
        if not self.image_input.connect():
            print("✗ Image loading failed")
            return False
        self.image_input.start_stream()
        
        print("\n[2/4] Loading segmentation model...")
        if not self.segmenter.load_model():
            print("✗ Model loading failed")
            return False
        
        print("\n[3/4] Verifying image input...")
        test_frame = self.image_input.get_frame()
        if test_frame is None:
            print("✗ Cannot get frames")
            return False
        
        print("\n[4/4] Checking robot connection...")
        self._check_robot_connection()
        
        print("\n✓ System setup complete!")
        return True
    
    def _check_robot_connection(self):
        """Check if Raspberry Pi robot is reachable."""
        try:
            result = subprocess.run(
                ['ping', '-n', '1', '-w', '1000', '10.42.0.1'],
                capture_output=True,
                timeout=2
            )
            
            if result.returncode == 0:
                print("  ✓ Robot Pi reachable at 10.42.0.1")
                print("  ✓ Commands will be sent via SCP")
                self.robot_connected = True
            else:
                print("  ⚠ Robot Pi not responding")
                print("  ⚠ Running in SIMULATION mode (no actual robot)")
                self.robot_connected = False
                
        except Exception as e:
            print(f"  ⚠ Connection check failed: {e}")
            print("  ⚠ Running in SIMULATION mode")
            self.robot_connected = False
    
    def set_navigation_goal(self, start_grid, goal_grid):
        """Set start and goal positions."""
        self.current_position_grid = start_grid
        self.goal_position = goal_grid
        
        print(f"\n✓ Algorithm: {self.current_algorithm}")
        print(f"Navigation goal set:")
        print(f"  Start: {start_grid}")
        print(f"  Goal: {goal_grid}")
        # Check if start and goal are free
        start=start_grid
        startx=start_grid[0]
        starty=start_grid[1]
        goal=goal_grid
        goalx=goal_grid[0]
        goaly=goal_grid[1]
        while not (self.map_builder.is_cell_free(start[0],start[1])):
            print(f"Start is NOT free, shifting goal to {(startx+2,starty+2)}")
            startx+=2
            starty+=2
            start = (startx, starty)
        print(f"Start is free: {self.map_builder.is_cell_free(start[0], start[1])}")   
        while not (self.map_builder.is_cell_free(goal[0],goal[1])):
            print(f"Goal is NOT free, shifting goal to {(goalx-2,goaly-2)}")
            goalx-=2
            goaly-=2
            goal = (goalx, goaly)
        print(f"Goal is free: {self.map_builder.is_cell_free(goal[0], goal[1])}")
    
    def run(self):
        """Main navigation loop."""
        print("\n" + "="*70)
        print("🤖 STARTING DYNAMIC NAVIGATION")
        print("="*70)
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  Q - Quit")
        print("  R - Force replan")
        print("\nOptimizations:")
        print("  ✓ Movements split into max 7-unit chunks")
        print("  ✓ Replanning only after moves (not after turns)")
        print("="*70)
        
        paused = False
        last_command_was_turn = False  # Track if we just turned
        
        try:
            while True:
                self.iteration += 1
                
                # Check keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠ User requested quit")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"\n{'⏸ PAUSED' if paused else '▶ RESUMED'}")
                    continue
                elif key == ord('r'):
                    print("\n🔄 Force replan requested")
                    last_command_was_turn = False  # Force replan
                
                if paused:
                    time.sleep(0.1)
                    continue
                
                # Print iteration header
                print(f"\n{'='*70}")
                print(f"ITERATION {self.iteration}")
                print(f"{'='*70}")
                print(f"Current position: {self.current_position_grid}")
                print(f"Current heading: {self.current_heading:.1f}°")
                print(f"Goal position: {self.goal_position}")
                
                # Calculate distance to goal
                if self.current_position_grid and self.goal_position:
                    dx = self.goal_position[0] - self.current_position_grid[0]
                    dy = self.goal_position[1] - self.current_position_grid[1]
                    dist_to_goal = np.sqrt(dx**2 + dy**2)
                    print(f"Distance to goal: {dist_to_goal:.1f} cells")
                    
                    # Check if goal reached
                    if dist_to_goal < 2:
                        print("\n" + "="*70)
                        print("🎉 GOAL REACHED!")
                        print("="*70)
                        break
                
                # OPTIMIZATION: Only replan after moves, not after turns
                # (Turning doesn't change position, so no need to rescan environment)
                if last_command_was_turn:
                    print("\n⏭ Skipping capture/replan (just turned, same position)")
                    # Don't capture new image or replan
                    # Just get the existing path and continue
                    path = self.current_path
                else:
                    # STEP 1: Capture and segment (with dynamic reload!)
                    if not self._capture_and_segment():
                        continue
                    
                    # STEP 2: Plan smooth 360° path
                    path = self._plan_path()
                    if not path:
                        print("⚠ Planning failed")
                        break
                    
                    # Store path for next iteration
                    self.current_path = path
                
                # STEP 3: Convert to smooth commands (split into max 7-unit moves)
                commands = self._convert_path(path)
                if not commands:
                    print("⚠ No commands generated")
                    break
                
                # STEP 4: Send FIRST command only
                if not self._send_first_command(commands):
                    continue
                
                # Track what type of command we just sent
                last_command_was_turn = commands[0].startswith('turn(')
                if last_command_was_turn:
                    print(f"  → Next iteration will skip replan (just turned)")
                else:
                    print(f"  → Next iteration will replan (moved to new position)")
                
                # STEP 5: Update robot state with FIXED coordinates
                self._update_robot_state(commands[0], path)
                
                # STEP 6: Visualize
                self._visualize_state(path)
                
                # Wait before next iteration
                print("⏳ Waiting before next iteration...")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n⚠ Interrupted")
        
        finally:
            self._print_summary()
    
    def _capture_and_segment(self):
        """Capture image and segment."""
        print("\n[Step 1/5] Capturing and segmenting image...")
        
        # Get frame (automatically reloads if file changed!)
        frame = self.image_input.get_frame()
        if frame is None:
            print("  ✗ No frame")
            return False
        
        self.current_image = frame
        
        # Segment
        mask = self.segmenter.segment(frame)
        if mask is None:
            print("  ✗ Segmentation failed")
            return False
        
        self.current_mask = mask
        
        # Update map
        self.map_builder.update_from_segmentation(mask)
        
        # Print stats
        stats = self.segmenter.get_statistics(mask)
        print(f"  Segmentation: {stats['safe_percentage']:.1f}% safe")
        
        return True
    
    def _plan_path(self):
        """Plan smooth 360° path."""
        print("\n[Step 2/5] Planning path using {self.current_algorithm}...")
        
        planner = self.planners[self.current_algorithm]
        path = planner.plan(self.current_position_grid, self.goal_position)
        
        if path:
            print(f"✓ Path planned: {len(path)} waypoints")
        
        return path
    
    def _convert_path(self, path):
        """Convert to smooth 360° commands."""
        print("\n[Step 3/5] Converting path to commands...")
        
        commands = self.path_converter.convert_path_to_commands(
            path, 
            start_heading=self.current_heading
        )
        commands = self.path_converter.optimize_commands(commands)
        
        print(f"✓ Generated {len(commands)} commands")
        
        return commands
    
    def _send_first_command(self, commands):
        """Send only first command to robot via SCP."""
        print("\n[Step 4/5] Sending first command to robot...")
        
        if not commands:
            return False
        
        first_cmd = commands[0]
        
        # Save to local file
        try:
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/robot_path.txt", 'w') as f:
                f.write(first_cmd + '\n')
        except Exception as e:
            print(f"  ✗ Local save error: {e}")
            return False
        
        # Send to Raspberry Pi via SCP
        try:
            import tempfile
            
            # Create temporary file with command
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(first_cmd + '\n')
                temp_path = temp_file.name
            
            # SCP to Pi
            remote_path = "asanku@10.42.0.1:/home/asanku/Documents/RSEF/movements/robot_path.txt"
            
            result = subprocess.run(
                ['scp', temp_path, remote_path],
                capture_output=True,
                timeout=5
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            if result.returncode == 0:
                print(f"✓ Sent to Pi: {first_cmd}")
                self.commands_sent.append(first_cmd)
                return True
            else:
                print(f"  ⚠ SCP failed (return code {result.returncode})")
                print(f"  Command saved locally, robot may not receive it")
                self.commands_sent.append(first_cmd)
                return True  # Continue anyway
                
        except subprocess.TimeoutExpired:
            print(f"  ⚠ SCP timeout (Pi not reachable?)")
            print(f"  Command: {first_cmd}")
            self.commands_sent.append(first_cmd)
            return True  # Continue in simulation mode
            
        except Exception as e:
            print(f"  ⚠ Send error: {str(e)}")
            print(f"  Command: {first_cmd}")
            self.commands_sent.append(first_cmd)
            return True  # Continue anyway
    
    def _update_robot_state(self, command, path):
        """
        Update robot state with FIXED coordinate system.
        Handles both 8-directional AND continuous angles!
        """
        print("\n[Step 5/5] Updating robot state...")
        
        if command.startswith('turn('):
            # Extract angle (can be float now!)
            angle_str = command[5:-1]
            angle = float(angle_str)
            self.current_heading = angle
            print(f"  Robot turned to {angle:.1f}°")
        
        elif command.startswith('move('):
            # Extract distance
            distance_str = command[5:-1]
            distance = float(distance_str)
            
            # Convert heading to movement vector (IMAGE coordinates!)
            # For 360° smooth paths, use trigonometry
            heading_rad = np.radians(self.current_heading)
            
            # CRITICAL: Image coordinates
            # 0° = East (right, +X)
            # 90° = South (down, +Y)
            # Standard trig works if we use this convention!
            dx = distance * np.cos(heading_rad)
            dy = distance * np.sin(heading_rad)
            
            # Update position
            old_pos = self.current_position_grid
            new_x = old_pos[0] + dx
            new_y = old_pos[1] + dy
            
            # Round to grid
            new_x = int(round(new_x))
            new_y = int(round(new_y))
            
            # Clamp
            new_x = max(0, min(new_x, self.map_builder.width - 1))
            new_y = max(0, min(new_y, self.map_builder.height - 1))
            
            self.current_position_grid = (new_x, new_y)
            
            print(f"  Robot moved {distance:.1f} cells at {self.current_heading:.1f}°")
            print(f"  Direction: dx={dx:+.1f}, dy={dy:+.1f}")
            print(f"  Position: {old_pos} → {self.current_position_grid}")
    
    def _visualize_state(self, path):
        """Create 4-window visualization."""
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
        
        # Window 3: Navigation map
        map_vis = self.map_builder.visualize(
            path=path,
            start=self.current_position_grid,
            goal=self.goal_position
        )
        cv2.putText(map_vis, f"Heading: {self.current_heading:.1f}°", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("3. Navigation Map", map_vis)
        
        # Window 4: Progress tracker
        tracker = np.zeros((400, 600, 3), dtype=np.uint8)
        tracker[:] = (40, 40, 40)
        
        y_pos = 40
        cv2.putText(tracker, f"ITERATION: {self.iteration}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        y_pos += 35
        # Show robot connection status
        if self.robot_connected:
            status_text = "Robot: CONNECTED"
            status_color = (0, 255, 0)
        else:
            status_text = "Robot: SIMULATION"
            status_color = (100, 100, 255)
        cv2.putText(tracker, status_text, (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        y_pos += 35
        cv2.putText(tracker, f"Position: {self.current_position_grid}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 30
        cv2.putText(tracker, f"Heading: {self.current_heading:.1f}°", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 30
        if self.current_position_grid and self.goal_position:
            dx = self.goal_position[0] - self.current_position_grid[0]
            dy = self.goal_position[1] - self.current_position_grid[1]
            dist = np.sqrt(dx**2 + dy**2)
            cv2.putText(tracker, f"Distance to goal: {dist:.1f}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Progress bar
            y_pos += 50
            total_dist = np.sqrt(
                (self.goal_position[0] - 10)**2 + 
                (self.goal_position[1] - 10)**2
            )
            progress = max(0, min(1, 1 - dist / total_dist))
            
            cv2.rectangle(tracker, (20, y_pos), (580, y_pos + 30), (100, 100, 100), 2)
            bar_width = int(560 * progress)
            cv2.rectangle(tracker, (20, y_pos), (20 + bar_width, y_pos + 30), (0, 255, 0), -1)
            
            cv2.putText(tracker, f"{progress*100:.0f}%", (270, y_pos + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 60
        cv2.putText(tracker, f"Commands sent: {len(self.commands_sent)}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Recent commands
        y_pos += 40
        cv2.putText(tracker, "Recent commands:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        for i, cmd in enumerate(self.commands_sent[-5:]):
            y_pos += 25
            cv2.putText(tracker, f"  {cmd}", (30, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        cv2.imshow("4. Progress Tracker", tracker)
    
    def _print_summary(self):
        """Print navigation summary."""
        print("\n" + "="*70)
        print("Navigation Summary:")
        print(f"  Iterations: {self.iteration}")
        print(f"  Commands sent: {len(self.commands_sent)}")
        print(f"  Final position: {self.current_position_grid}")
        print("="*70)
        print("Press any key to close visualization windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\nShutting down...")
        self.image_input.disconnect()
        print("✓ Shutdown complete")


def main():
    """Main entry point."""
    # Create system
    system = DynamicNavigationSystem()
    
    # Setup
    if not system.setup():
        print("✗ Setup failed")
        return 1
    start = (10, 10)
    goal = (90, 90)
    system.set_navigation_goal(start, goal)
    
    # Run navigation
    system.run()
    
    return 0


if __name__ == "__main__":
    config.create_directories()
    exit_code = main()
    sys.exit(exit_code)
