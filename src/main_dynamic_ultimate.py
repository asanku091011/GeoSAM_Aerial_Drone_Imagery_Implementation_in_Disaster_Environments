"""
ULTIMATE Dynamic Navigation System - FINAL
Square obstacle marking 20 units ahead
Matches robot dot visualization size
No backup (BACKUP_DISTANCE = 0)
Position tracking fixed
Interactive mode selection at startup
"""

import sys
import time
import cv2
import numpy as np
import os
import subprocess
import json

os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import config

from dynamic_image_input import DynamicImageInput
from proper_sam_segmenter import AccurateSAMSegmenter
from mobilesam_segmenter import MobileSAMSegmenter
from map_builder import MapBuilder
from astar import AStarPlanner
from rrt_star import RRTStarPlanner
from greedy import GreedyPlanner
from path_converter_smooth import SmoothPathConverter
from data_logger import DataLogger


class DynamicNavigationSystem:
    
    ROBOT_WIDTH_CELLS = 10
    BACKUP_DISTANCE = 0  # No backup
    
    def __init__(self):
        print("\n" + "="*70)
        print("ULTIMATE DYNAMIC NAVIGATION SYSTEM - FINAL")
        print("="*70)
        print("Features:")
        print("  Static mode: plan ONCE, follow until done")
        print("  Ultrasonic: marks obstacle square 20 units ahead")
        print("  Smooth paths: minimum 5-unit movements")
        print("="*70)
        print("\nInitializing components...\n")
        
        self.image_input = DynamicImageInput("data/test_images/"+config.TEST_IMAGE+".jpg")
        
        print("="*70)
        print("PRELOADING SEGMENTATION MODELS")
        print("="*70)
        
        print("\n[1/2] Preloading Full SAM...")
        self.sam_segmenter = AccurateSAMSegmenter()
        sam_load_start = time.time()
        self.sam_segmenter.load_model()
        sam_load_time = time.time() - sam_load_start
        print(f"  Full SAM ready ({sam_load_time:.1f}s)")
        
        print("\n[2/2] Preloading MobileSAM...")
        self.mobilesam_segmenter = MobileSAMSegmenter()
        mobile_load_start = time.time()
        self.mobilesam_segmenter.load_model()
        mobile_load_time = time.time() - mobile_load_start
        print(f"  MobileSAM ready ({mobile_load_time:.1f}s)")
        
        # Set which model we're using based on config
        if config.SEGMENTATION_MODEL == 'sam':
            self.segmenter = self.sam_segmenter
            self.current_model = 'sam'
        elif config.SEGMENTATION_MODEL == 'mobilesam':
            self.segmenter = self.mobilesam_segmenter
            self.current_model = 'mobilesam'
        else:
            self.segmenter = config.get_segmenter()
            self.current_model = config.SEGMENTATION_MODEL
        
        total_load_time = sam_load_time + mobile_load_time
        print("="*70)
        print(f"ALL MODELS PRELOADED! Total: {total_load_time:.1f}s")
        print("="*70)
        
        # Create the map builder for navigation
        self.map_builder = MapBuilder()
        self.path_converter = SmoothPathConverter(self.map_builder, unit_scale=1.0)
        
        # Make sure all output folders exist
        config.create_directories()
        self.logger = DataLogger()
        
        # Set up all three path planning algorithms
        self.planners = {
            'astar': AStarPlanner(self.map_builder),
            'rrt_star': RRTStarPlanner(self.map_builder),
            'greedy': GreedyPlanner(self.map_builder)
        }
        
        # Robot state variables
        self.current_algorithm = config.DEFAULT_ALGORITHM
        self.current_position_grid = None
        self.current_heading = 0.0
        self.goal_position = None
        self.iteration = 0
        self.commands_sent = []
        self.robot_connected = False
        
        # Path planning state
        self.planned_path = None
        self.remaining_commands = []
        self.path_planned = False
        
        # Ultrasonic sensor tracking
        self.last_ultrasonic_distance = None
        self.obstacle_detected_count = 0
        self.ultrasonic_history = []
        self.detected_obstacles = set()  # Keeps track of obstacles we already marked
        self.permanent_obstacles = []  # Stores obstacle positions so they don't disappear when we resegment
        
        # Visualization window sizes
        self.vis_width = 640
        self.vis_height = 480
        
        print("\nAll components initialized\n")
    
    def switch_segmenter(self, model_type):
        """Switch between SAM and MobileSAM models during runtime"""
        if model_type == 'sam' and self.current_model != 'sam':
            self.segmenter = self.sam_segmenter
            self.current_model = 'sam'
            return True
        elif model_type == 'mobilesam' and self.current_model != 'mobilesam':
            self.segmenter = self.mobilesam_segmenter
            self.current_model = 'mobilesam'
            return True
        return False
    
    def setup(self):
        """Initialize the system and check connections"""
        print("Setting up system...")
        
        print("\n[1/3] Loading dynamic image...")
        if not self.image_input.connect():
            return False
        self.image_input.start_stream()
        
        print("\n[2/3] Verifying image input...")
        test_frame = self.image_input.get_frame()
        if test_frame is None:
            return False
        
        print("\n[3/3] Checking robot connection...")
        self._check_robot_connection()
        
        print("\nSystem setup complete!")
        return True
    
    def _check_robot_connection(self):
        """Check if the Raspberry Pi robot is connected on the network"""
        try:
            result = subprocess.run(
                ['ping', '-n', '1', '-w', '1000', '10.42.0.1'],
                capture_output=True,
                timeout=2
            )
            
            if result.returncode == 0:
                print("  Robot Pi reachable")
                self.robot_connected = True
            else:
                print("  SIMULATION mode")
                self.robot_connected = False
        except:
            self.robot_connected = False
    
    def _fetch_robot_status(self):
        """Get the current status from the robot including ultrasonic sensor data"""
        if not self.robot_connected:
            return None
        
        try:
            import tempfile
            
            # Create a temporary file to download the status into
            with tempfile.NamedTemporaryFile(mode='r', delete=False, suffix='.json') as temp_file:
                temp_path = temp_file.name
            
            # Path to robot status file on the Pi
            remote_status = "asanku@10.42.0.1:/home/asanku/Documents/RSEF/status/robot_status.json"
            
            # Download the file using SCP
            result = subprocess.run(
                ['scp', remote_status, temp_path],
                capture_output=True,
                timeout=2
            )
            
            if result.returncode == 0:
                with open(temp_path, 'r') as f:
                    status = json.load(f)
                os.unlink(temp_path)
                return status
            else:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None
        except:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            return None
    
    def _handle_ultrasonic_obstacle(self, status):
        """
        Handle when the ultrasonic sensor detects an obstacle
        Marks a square obstacle on the map 20 units ahead of the robot
        """
        ultrasonic_cm = status.get('ultrasonic_cm')
        obstacle_detected = status.get('obstacle_detected', False)
        
        # Keep track of ultrasonic readings
        self.last_ultrasonic_distance = ultrasonic_cm
        if ultrasonic_cm is not None:
            self.ultrasonic_history.append(ultrasonic_cm)
            if len(self.ultrasonic_history) > 10:
                self.ultrasonic_history.pop(0)
        
        if not obstacle_detected:
            return False
        
        print("\n" + "="*70)
        print("ULTRASONIC OBSTACLE DETECTED!")
        print("="*70)
        print(f"  Distance: {ultrasonic_cm:.1f}cm")
        print(f"  Robot heading: {self.current_heading:.1f} degrees")
        print(f"  Robot position: {self.current_position_grid}")
        print("="*70)
        
        # Convert heading to radians for math
        heading_rad = np.radians(self.current_heading)
        
        # Settings for obstacle marking
        OBSTACLE_DISTANCE_AHEAD = 20  # How far ahead to mark the obstacle
        OBSTACLE_SIZE = 8  # Size of the obstacle square (matches robot visualization)
        
        # Calculate where the obstacle center should be (20 units ahead)
        obstacle_center_x = self.current_position_grid[0] + OBSTACLE_DISTANCE_AHEAD * np.cos(heading_rad)
        obstacle_center_y = self.current_position_grid[1] + OBSTACLE_DISTANCE_AHEAD * np.sin(heading_rad)
        
        # Round to nearest cell
        obstacle_center_x = int(round(obstacle_center_x))
        obstacle_center_y = int(round(obstacle_center_y))
        
        # Make sure it's inside the map
        obstacle_center_x = max(0, min(obstacle_center_x, self.map_builder.width - 1))
        obstacle_center_y = max(0, min(obstacle_center_y, self.map_builder.height - 1))
        
        # Check if we already marked an obstacle in this area
        obstacle_key = (obstacle_center_x // 10, obstacle_center_y // 10)
        if obstacle_key in self.detected_obstacles:
            print(f"  Already marked, skipping")
            return False
        
        # Remember that we marked this area
        self.detected_obstacles.add(obstacle_key)
        self.obstacle_detected_count += 1
        
        print(f"\nMARKING OBSTACLE SQUARE:")
        print(f"  Center: ({obstacle_center_x}, {obstacle_center_y})")
        print(f"  Distance ahead: {OBSTACLE_DISTANCE_AHEAD} units")
        print(f"  Size: {OBSTACLE_SIZE}x{OBSTACLE_SIZE} cells")
        
        # Mark a square region around the obstacle center
        cells_marked = 0
        half_size = OBSTACLE_SIZE
        
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                mark_x = obstacle_center_x + dx
                mark_y = obstacle_center_y + dy
                
                # Make sure we're inside the map
                if (0 <= mark_x < self.map_builder.width and 
                    0 <= mark_y < self.map_builder.height):
                    # Mark as obstacle (0 = blocked, 1 = free)
                    self.map_builder.grid[mark_y, mark_x] = 0
                    cells_marked += 1
                    
                    # Also update the mask if we have one
                    if hasattr(self, 'current_mask'):
                        self.current_mask[mark_y, mark_x] = 0
        
        print(f"  Marked {cells_marked} cells")
        
        # Store this obstacle permanently so it doesn't disappear when we resegment
        obstacle_cells = []
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                mark_x = obstacle_center_x + dx
                mark_y = obstacle_center_y + dy
                
                if (0 <= mark_x < self.map_builder.width and 
                    0 <= mark_y < self.map_builder.height):
                    obstacle_cells.append((mark_x, mark_y))
        
        # Save all the info about this obstacle
        self.permanent_obstacles.append({
            'cells': obstacle_cells,
            'center': (obstacle_center_x, obstacle_center_y),
            'size': OBSTACLE_SIZE
        })
        
        print(f"  Stored {len(obstacle_cells)} cells for permanent obstacle #{self.obstacle_detected_count}")
        
        # Robot doesn't backup since BACKUP_DISTANCE = 0
        
        print(f"\nFORCING REPLAN!")
        
        return True
    
    def _reapply_permanent_obstacles(self):
        """
        Put all the ultrasonic obstacles back on the map after resegmentation
        This makes sure obstacles don't disappear when we get a new image
        """
        if not self.permanent_obstacles:
            return
        
        total_cells = 0
        for obstacle in self.permanent_obstacles:
            for cell_x, cell_y in obstacle['cells']:
                if (0 <= cell_x < self.map_builder.width and 
                    0 <= cell_y < self.map_builder.height):
                    self.map_builder.grid[cell_y, cell_x] = 0
                    total_cells += 1
                    
                    if hasattr(self, 'current_mask'):
                        self.current_mask[cell_y, cell_x] = 0
        
        if total_cells > 0:
            print(f"  Reapplied {len(self.permanent_obstacles)} permanent obstacles ({total_cells} cells)")
    
    def _send_backup_command(self):
        """Not used since BACKUP_DISTANCE = 0"""
        return True
    
    def set_navigation_goal(self, start_grid, goal_grid):
        """Set where the robot starts and where it needs to go"""
        self.current_position_grid = start_grid
        self.goal_position = goal_grid
        
        print(f"\nAlgorithm: {self.current_algorithm}")
        print(f"Navigation goal:")
        print(f"  Start: {start_grid}")
        print(f"  Goal: {goal_grid}")
    
    def _validate_and_shift_positions(self):
        """
        Make sure start and goal positions are in free space
        If they're on an obstacle, shift them slightly
        """
        print("\n[Validating positions]")
        
        # Check start position
        startx, starty = self.current_position_grid
        original_start = self.current_position_grid
        
        # If start is blocked, keep shifting it until we find free space
        while not self.map_builder.is_cell_free(startx, starty):
            startx += 2
            starty += 2
            if startx >= self.map_builder.width or starty >= self.map_builder.height:
                return False
        
        self.current_position_grid = (startx, starty)
        
        if (startx, starty) != original_start:
            print(f"  Start: {original_start} -> {(startx, starty)}")
        else:
            print(f"  Start: {(startx, starty)}")
        
        # Check goal position
        goalx, goaly = self.goal_position
        original_goal = self.goal_position
        
        # If goal is blocked, keep shifting it until we find free space
        while not self.map_builder.is_cell_free(goalx, goaly):
            goalx -= 2
            goaly -= 2
            if goalx < 0 or goaly < 0:
                return False
        
        self.goal_position = (goalx, goaly)
        
        if (goalx, goaly) != original_goal:
            print(f"  Goal: {original_goal} -> {(goalx, goaly)}")
        else:
            print(f"  Goal: {(goalx, goaly)}")
        
        return True

    def run(self):
        """Main navigation loop - this is where everything happens"""
        print("\n" + "="*70)
        print("STATIC MODE NAVIGATION")
        print("="*70)
        print("Controls: SPACE=pause Q=quit R=replan 1=SAM 2=MobileSAM")
        print("Behavior: Plan once -> Execute -> Only replan if obstacle")
        print("="*70)
        
        paused = False
        positions_validated = False
        initial_segmentation_done = False
        
        try:
            while True:
                self.iteration += 1
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    continue
                elif key == ord('r'):
                    # Force a replan
                    self.path_planned = False
                    self.remaining_commands = []
                elif key == ord('1'):
                    # Switch to full SAM
                    if self.switch_segmenter('sam'):
                        self.path_planned = False
                        positions_validated = False
                elif key == ord('2'):
                    # Switch to MobileSAM
                    if self.switch_segmenter('mobilesam'):
                        self.path_planned = False
                        positions_validated = False
                
                if paused:
                    time.sleep(0.1)
                    continue
                
                print(f"\n{'='*70}")
                print(f"ITERATION {self.iteration}")
                print(f"{'='*70}")
                print(f"Pos: {self.current_position_grid} | Head: {self.current_heading:.1f} deg | Goal: {self.goal_position}")
                
                # Check if ultrasonic sensor detected anything
                robot_status = self._fetch_robot_status()
                if robot_status:
                    if self._handle_ultrasonic_obstacle(robot_status):
                        # Obstacle detected! Need to replan
                        self.path_planned = False
                        self.remaining_commands = []
                
                # Do the initial segmentation (only happens once)
                if not initial_segmentation_done:
                    print("\nInitial segmentation...")
                    if not self._capture_and_segment():
                        continue
                    
                    initial_segmentation_done = True
                    
                    if not positions_validated:
                        if not self._validate_and_shift_positions():
                            print("Invalid positions")
                            break
                        positions_validated = True
                
                # DYNAMIC MODE: Resegment every iteration if dynamic mode is on
                elif not config.STATIC_IMAGE_MODE:
                    print("\nDynamic resegmentation...")
                    if not self._capture_and_segment():
                        continue
                    
                    # Put ultrasonic obstacles back on the new segmentation
                    self._reapply_permanent_obstacles()
                    
                    # Force a replan with the updated map
                    self.path_planned = False
                    self.remaining_commands = []
                    print("  -> Forcing replan after resegmentation")
                
                # Plan a path if we don't have one yet
                if not self.path_planned or not self.remaining_commands:
                    print("\nPlanning...")
                    
                    # If this is a replan, check positions again
                    if self.path_planned:
                        if not self._validate_and_shift_positions():
                            print("Cannot replan")
                            break
                    
                    path = self._plan_path()
                    if not path:
                        print("No path")
                        break
                    
                    self.planned_path = path
                    
                    # Convert path to movement commands
                    all_commands = self._convert_path(path)
                    if not all_commands:
                        print("No commands")
                        break
                    
                    self.remaining_commands = all_commands.copy()
                    self.path_planned = True
                    
                    print(f"{len(self.remaining_commands)} commands queued")
                
                # Check if we reached the goal
                if self.current_position_grid and self.goal_position:
                    dx = self.goal_position[0] - self.current_position_grid[0]
                    dy = self.goal_position[1] - self.current_position_grid[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < 5:
                        print("\nGOAL REACHED!")
                        break
                
                # Execute the next command
                if not self.remaining_commands:
                    print("\nNo commands left, replanning")
                    self.path_planned = False
                    continue
                
                cmd = self.remaining_commands.pop(0)
                print(f"\n-> {cmd} ({len(self.remaining_commands)} left)")
                
                self._send_command(cmd)
                self._update_robot_state(cmd, self.planned_path)
                self._visualize_state(self.planned_path)
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        finally:
            self._print_summary()
    
    def _capture_and_segment(self):
        """Get an image and segment it into safe/unsafe regions"""
        frame = self.image_input.get_frame()
        if frame is None:
            return False
        
        self.current_image = frame
        
        # Run segmentation on the image
        mask = self.segmenter.segment(frame)
        if mask is None:
            return False
        
        self.current_mask = mask
        self.map_builder.update_from_segmentation(mask)
        
        # Print how much of the map is safe
        stats = self.segmenter.get_statistics(mask)
        print(f"  {stats['safe_percentage']:.1f}% safe")
        
        return True
    
    def _plan_path(self):
        """Use A* to plan a path from current position to goal"""
        planner = self.planners[self.current_algorithm]
        path = planner.plan(self.current_position_grid, self.goal_position)
        
        if path:
            print(f"  {len(path)} waypoints")
        
        return path
    
    def _convert_path(self, path):
        """Convert the path into turn() and move() commands"""
        commands = self.path_converter.convert_path_to_commands(
            path, start_heading=self.current_heading
        )
        commands = self.path_converter.optimize_commands(commands)
        
        return commands
    
    def _send_command(self, command):
        """Send a command to the robot via SCP"""
        try:
            # Save to local outputs folder
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/robot_path.txt", 'w') as f:
                f.write(command + '\n')
        except:
            pass
        
        if not self.robot_connected:
            self.commands_sent.append(command)
            return True
        
        try:
            import tempfile
            
            # Create temp file with command
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(command + '\n')
                temp_path = temp_file.name
            
            # Upload to robot using SCP
            remote_path = "asanku@10.42.0.1:/home/asanku/Documents/RSEF/movements/robot_path.txt"
            
            result = subprocess.run(
                ['scp', temp_path, remote_path],
                capture_output=True,
                timeout=5
            )
            
            os.unlink(temp_path)
            
            self.commands_sent.append(command)
            return True
        except:
            self.commands_sent.append(command)
            return True
    
    def _update_robot_state(self, command, path):
        """Update where we think the robot is based on the command we sent"""
        if command.startswith('turn('):
            # Update heading
            angle = float(command[5:-1])
            self.current_heading = angle
        
        elif command.startswith('move('):
            # Update position
            distance = float(command[5:-1])
            
            heading_rad = np.radians(self.current_heading)
            
            # Calculate how much to move in x and y
            dx = distance * np.cos(heading_rad)
            dy = distance * np.sin(heading_rad)
            
            old_pos = self.current_position_grid
            new_x = int(round(old_pos[0] + dx))
            new_y = int(round(old_pos[1] + dy))
            
            # Make sure we stay inside the map
            new_x = max(0, min(new_x, self.map_builder.width - 1))
            new_y = max(0, min(new_y, self.map_builder.height - 1))
            
            self.current_position_grid = (new_x, new_y)
    
    def _visualize_state(self, path):
        """Show visualization windows with current state"""
        # Window 1: Original image
        if hasattr(self, 'current_image'):
            img_vis = cv2.resize(self.current_image, (self.vis_width, self.vis_height))
            cv2.putText(img_vis, f"Iter {self.iteration}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("1. Image", img_vis)
        
        # Window 2: Segmentation overlay
        if hasattr(self, 'current_mask'):
            seg_vis = self.segmenter.visualize_segmentation(
                self.current_image, self.current_mask
            )
            seg_vis = cv2.resize(seg_vis, (self.vis_width, self.vis_height))
            cv2.imshow("2. Segmentation", seg_vis)
        
        # Window 3: Navigation map with path
        map_vis = self.map_builder.visualize(
            path=path,
            start=self.current_position_grid,
            goal=self.goal_position
        )
        map_vis = cv2.resize(map_vis, (self.vis_width, self.vis_height), 
                            interpolation=cv2.INTER_NEAREST)
        
        cv2.putText(map_vis, f"Head: {self.current_heading:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show ultrasonic distance (red if close, green if far)
        if self.last_ultrasonic_distance is not None:
            ultra_color = (0, 0, 255) if self.last_ultrasonic_distance < 20 else (0, 255, 0)
            cv2.putText(map_vis, f"{self.last_ultrasonic_distance:.1f}cm", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ultra_color, 2)
        
        cv2.imshow("3. Map", map_vis)
        
        # Window 4: Progress tracker
        tracker = np.zeros((450, 600, 3), dtype=np.uint8)
        tracker[:] = (40, 40, 40)
        
        y = 40
        cv2.putText(tracker, f"ITER: {self.iteration}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        y += 35
        cv2.putText(tracker, f"Queue: {len(self.remaining_commands)}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        y += 35
        cv2.putText(tracker, f"Obstacles: {self.obstacle_detected_count}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
        
        cv2.imshow("4. Progress", tracker)
    
    def _print_summary(self):
        """Print final statistics when navigation ends"""
        print("\n" + "="*70)
        print("Summary:")
        print(f"  Iterations: {self.iteration}")
        print(f"  Commands: {len(self.commands_sent)}")
        print(f"  Obstacles: {self.obstacle_detected_count}")
        print("="*70)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        self.image_input.disconnect()


def select_mode():
    """
    Ask the user which mode they want to run
    This makes it easier than editing config.py every time
    """
    print("\n" + "="*70)
    print("MODE SELECTION")
    print("="*70)
    print("\nAvailable modes:")
    print("  1. Static + testearthquake + MobileSAM (STATIC WITH EARTHQUAKE)")
    print("  2. Dynamic + current_scene + MobileSAM (DYANMIC WITH IMAGE GENERATOR)")
    print("  3. Static + DJIDrone + MobileSAM (STATIC WITH DRONE IMAGERY)")
    print("  4. Static + testearthquake + Full SAM (STATIC EARTHQUAKE WITH FULL SAM MODEL)")
    print("  5. Custom (use current config.py settings)")
    print("="*70)
    
    while True:
        try:
            choice = input("\nSelect mode [1-5]: ").strip()
            
            if choice == '1':
                config.STATIC_IMAGE_MODE = True
                config.TEST_IMAGE = "testearthquake"
                config.SEGMENTATION_MODEL = 'mobilesam'
                print("\nMode 1: Static + testearthquake + MobileSAM")
                break
            elif choice == '2':
                config.STATIC_IMAGE_MODE = False
                config.TEST_IMAGE = "current_scene"
                config.SEGMENTATION_MODEL = 'mobilesam'
                print("\nMode 2: Dynamic + current_scene + MobileSAM")
                break
            elif choice == '3':
                config.STATIC_IMAGE_MODE = True
                config.TEST_IMAGE = "DJIDrone"
                config.SEGMENTATION_MODEL = 'mobilesam'
                print("\nMode 3: Static + DJIDrone + MobileSAM")
                break
            elif choice == '4':
                config.STATIC_IMAGE_MODE = True
                config.TEST_IMAGE = "testearthquake"
                config.SEGMENTATION_MODEL = 'sam'
                print("\nMode 4: Static + testearthquake + Full SAM")
                break
            elif choice == '5':
                print("\nMode 5: Using config.py settings")
                print(f"  STATIC_IMAGE_MODE = {config.STATIC_IMAGE_MODE}")
                print(f"  TEST_IMAGE = {config.TEST_IMAGE}")
                print(f"  SEGMENTATION_MODEL = {config.SEGMENTATION_MODEL}")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\n\nCancelled")
            sys.exit(0)
        except:
            print("Invalid input. Please enter 1-5.")
    
    print(f"\nFinal configuration:")
    print(f"  Image: data/test_images/{config.TEST_IMAGE}.jpg")
    print(f"  Mode: {'Static (segment once)' if config.STATIC_IMAGE_MODE else 'Dynamic (resegment every iteration)'}")
    print(f"  Model: {config.SEGMENTATION_MODEL.upper()}")


def main():
    # Ask user which mode they want before loading anything
    select_mode()
    
    system = DynamicNavigationSystem()
    
    if not system.setup():
        return 1
    
    # Set start and goal positions
    start = (75, 75)
    goal = (config.MAP_WIDTH-75, config.MAP_HEIGHT-75)
    system.set_navigation_goal(start, goal)
    
    system.run()
    
    return 0


if __name__ == "__main__":
    config.create_directories()
    exit_code = main()
    sys.exit(exit_code)