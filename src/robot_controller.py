"""
Robot controller that converts planned paths into executable movement commands.
Designed for differential-drive robots running on Raspberry Pi 5.
"""

import numpy as np
import json
import time
import config

class RobotController:
    """
    Converts path waypoints into robot movement commands.
    
    This controller:
    - Takes a path from the planner
    - Converts grid coordinates to real-world positions
    - Generates movement commands (linear and angular velocities)
    - Outputs commands in a format ready for the robot hardware
    
    The controller is designed for differential-drive robots where:
    - Two wheels control motion
    - Commands specify linear velocity (forward/backward) and angular velocity (turning)
    """
    
    def __init__(self, map_builder):
        """
        Initialize robot controller.
        
        Args:
            map_builder: MapBuilder instance for coordinate conversion
        """
        self.map_builder = map_builder
        
        # Robot parameters
        self.wheel_base = config.ROBOT_WHEEL_BASE
        self.max_linear_speed = config.ROBOT_MAX_LINEAR_SPEED
        self.max_angular_speed = config.ROBOT_MAX_ANGULAR_SPEED
        
        # Current state
        self.current_position = None  # (x, y) in world coordinates
        self.current_heading = 0.0  # radians
        self.target_waypoint_idx = 0
        self.current_path = None
        
        # Command history
        self.command_history = []
        
        print("Robot Controller initialized")
        print(f"  Max linear speed: {self.max_linear_speed} m/s")
        print(f"  Max angular speed: {self.max_angular_speed} rad/s")
    
    def set_path(self, path):
        """
        Set a new path for the robot to follow.
        
        Args:
            path (list): Path as list of (x, y) tuples in grid coordinates
        """
        self.current_path = path
        self.target_waypoint_idx = 0
        print(f"New path set with {len(path)} waypoints")
    
    def set_position(self, position, heading=None):
        """
        Update the robot's current position and heading.
        
        Args:
            position (tuple): Position (x, y) in world coordinates (meters)
            heading (float): Heading angle in radians (optional)
        """
        self.current_position = position
        if heading is not None:
            self.current_heading = heading
    
    def update(self):
        """
        Generate the next movement command based on current state.
        
        Returns:
            dict: Movement command with 'linear' and 'angular' velocities,
                  or None if path is complete
        """
        # Check if we have a valid path
        if not self.current_path or self.target_waypoint_idx >= len(self.current_path):
            return None
        
        # Get target waypoint in world coordinates
        target_grid = self.current_path[self.target_waypoint_idx]
        target_world = self.map_builder.grid_to_world(target_grid[0], target_grid[1])
        
        # Calculate vector to target
        dx = target_world[0] - self.current_position[0]
        dy = target_world[1] - self.current_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Check if we reached this waypoint
        if distance < config.WAYPOINT_REACHED_THRESHOLD:
            self.target_waypoint_idx += 1
            
            # If that was the last waypoint, we're done
            if self.target_waypoint_idx >= len(self.current_path):
                return {'linear': 0.0, 'angular': 0.0, 'status': 'goal_reached'}
            
            # Move to next waypoint
            return self.update()
        
        # Calculate desired heading to target
        target_heading = np.arctan2(dy, dx)
        
        # Calculate heading error
        heading_error = self._normalize_angle(target_heading - self.current_heading)
        
        # Generate command using simple proportional control
        command = self._generate_command(distance, heading_error)
        
        # Store command in history
        self.command_history.append({
            'timestamp': time.time(),
            'command': command,
            'position': self.current_position,
            'target': target_world
        })
        
        return command
    
    def _generate_command(self, distance, heading_error):
        """
        Generate movement command using proportional control.
        
        Args:
            distance (float): Distance to target waypoint
            heading_error (float): Angle error to target
            
        Returns:
            dict: Command with linear and angular velocities
        """
        # Proportional gains
        K_linear = 0.5  # Linear velocity gain
        K_angular = 2.0  # Angular velocity gain
        
        # If heading error is large, prioritize turning
        if abs(heading_error) > np.pi / 6:  # More than 30 degrees off
            linear_velocity = 0.0  # Stop moving forward while turning
            angular_velocity = K_angular * heading_error
        else:
            # Move forward while adjusting heading
            linear_velocity = K_linear * distance
            angular_velocity = K_angular * heading_error
        
        # Apply velocity limits
        linear_velocity = np.clip(linear_velocity, 
                                 -self.max_linear_speed, 
                                 self.max_linear_speed)
        angular_velocity = np.clip(angular_velocity,
                                  -self.max_angular_speed,
                                  self.max_angular_speed)
        
        return {
            'linear': float(linear_velocity),
            'angular': float(angular_velocity),
            'status': 'navigating'
        }
    
    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi] range.
        
        Args:
            angle (float): Angle in radians
            
        Returns:
            float: Normalized angle
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def convert_to_wheel_speeds(self, command):
        """
        Convert linear and angular velocities to wheel speeds.
        Useful for differential-drive robots.
        
        Args:
            command (dict): Command with 'linear' and 'angular' velocities
            
        Returns:
            dict: Left and right wheel speeds in m/s
        """
        v = command['linear']  # Linear velocity
        omega = command['angular']  # Angular velocity
        
        # Differential drive kinematics
        v_left = v - (omega * self.wheel_base / 2.0)
        v_right = v + (omega * self.wheel_base / 2.0)
        
        return {
            'left_wheel': float(v_left),
            'right_wheel': float(v_right)
        }
    
    def save_commands(self, filename=None):
        """
        Save all commands to a file for robot execution.
        
        Args:
            filename (str): Output filename (default from config)
            
        Returns:
            bool: True if save successful
        """
        if filename is None:
            filename = config.COMMAND_OUTPUT_FILE
        
        try:
            # Create output directory if needed
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            if config.COMMAND_FORMAT == 'json':
                # Save as JSON
                with open(filename, 'w') as f:
                    json.dump(self.command_history, f, indent=2)
            else:
                # Save as simple text format
                with open(filename, 'w') as f:
                    f.write("# Robot Movement Commands\n")
                    f.write("# Format: timestamp, linear_vel, angular_vel\n\n")
                    
                    for entry in self.command_history:
                        cmd = entry['command']
                        f.write(f"{entry['timestamp']:.3f}, "
                               f"{cmd['linear']:.4f}, "
                               f"{cmd['angular']:.4f}\n")
            
            print(f"✓ Commands saved to {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save commands: {str(e)}")
            return False
    
    def get_raspberry_pi_code(self):
        """
        Generate Python code that can run on Raspberry Pi to execute commands.
        This code assumes you have motor control libraries installed.
        
        Returns:
            str: Python code for Raspberry Pi
        """
        code = '''#!/usr/bin/env python3
"""
Robot movement executor for Raspberry Pi 5
Auto-generated by Disaster Navigation System
"""

import time
import json

# Import your motor control library here
# For example: from gpiozero import Robot
# or: import RPi.GPIO as GPIO

class RobotExecutor:
    def __init__(self):
        # Initialize your motor controllers here
        # Example: self.robot = Robot(left=(17, 18), right=(22, 23))
        pass
    
    def set_velocities(self, linear, angular):
        """
        Set robot velocities.
        
        Args:
            linear (float): Forward velocity in m/s
            angular (float): Rotation velocity in rad/s
        """
        # Convert to wheel speeds
        wheel_base = {wheel_base}
        v_left = linear - (angular * wheel_base / 2.0)
        v_right = linear + (angular * wheel_base / 2.0)
        
        # Convert to motor PWM values (0-100)
        # Adjust these scaling factors for your robot
        max_speed = {max_linear_speed}
        left_pwm = int((v_left / max_speed) * 100)
        right_pwm = int((v_right / max_speed) * 100)
        
        # Send to motors
        # Example: self.robot.value = (left_pwm/100, right_pwm/100)
        print(f"Motors: L={left_pwm}%, R={right_pwm}%")
    
    def execute_commands(self, filename):
        """Load and execute commands from file."""
        with open(filename, 'r') as f:
            commands = json.load(f)
        
        print(f"Executing {{len(commands)}} commands...")
        
        for i, entry in enumerate(commands):
            cmd = entry['command']
            print(f"Step {{i+1}}/{{len(commands)}}: "
                  f"linear={{cmd['linear']:.3f}}, angular={{cmd['angular']:.3f}}")
            
            self.set_velocities(cmd['linear'], cmd['angular'])
            
            # Wait for update interval
            time.sleep(1.0 / {update_rate})
        
        # Stop at the end
        self.set_velocities(0, 0)
        print("Navigation complete!")

if __name__ == "__main__":
    executor = RobotExecutor()
    executor.execute_commands("{command_file}")
'''.format(
            wheel_base=self.wheel_base,
            max_linear_speed=self.max_linear_speed,
            update_rate=config.CONTROLLER_UPDATE_RATE,
            command_file=config.COMMAND_OUTPUT_FILE
        )
        
        return code
    
    def save_raspberry_pi_code(self, filename="outputs/robot_executor.py"):
        """
        Save the Raspberry Pi execution code to a file.
        
        Args:
            filename (str): Output filename
            
        Returns:
            bool: True if save successful
        """
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            code = self.get_raspberry_pi_code()
            
            with open(filename, 'w') as f:
                f.write(code)
            
            print(f"✓ Raspberry Pi code saved to {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save Raspberry Pi code: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    print("Testing Robot Controller Module")
    print("=" * 50)
    
    from map_builder import MapBuilder
    
    # Create map builder
    map_builder = MapBuilder()
    
    # Create controller
    controller = RobotController(map_builder)
    
    # Set up a simple path
    path = [(10, 10), (20, 10), (20, 20), (30, 20)]
    controller.set_path(path)
    
    # Set initial position
    start_world = map_builder.grid_to_world(10, 10)
    controller.set_position(start_world, heading=0.0)
    
    print("\nSimulating robot navigation...")
    print("=" * 50)
    
    # Simulate navigation
    for step in range(100):
        command = controller.update()
        
        if command is None:
            print("✓ Path complete!")
            break
        
        print(f"Step {step}: linear={command['linear']:.3f}, "
              f"angular={command['angular']:.3f}, status={command['status']}")
        
        # Simulate robot movement (in real system, this comes from odometry)
        dt = 1.0 / config.CONTROLLER_UPDATE_RATE
        controller.current_position = (
            controller.current_position[0] + command['linear'] * dt * np.cos(controller.current_heading),
            controller.current_position[1] + command['linear'] * dt * np.sin(controller.current_heading)
        )
        controller.current_heading += command['angular'] * dt
        
        time.sleep(0.1)  # Slow down for visualization
    
    # Save commands
    controller.save_commands()
    
    # Generate Raspberry Pi code
    controller.save_raspberry_pi_code()
    
    print("\n✓ Test complete!")