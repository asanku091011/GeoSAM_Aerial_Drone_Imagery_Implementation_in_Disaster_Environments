"""
Enhanced Path Converter for 360-degree smooth paths
Converts smoothed paths to robot commands with continuous angles
"""

import numpy as np
import math
import config


class SmoothPathConverter:
    """
    Converts smoothed paths to robot commands with 360-degree angles.
    
    Instead of just 8 directions, this supports ANY angle (0-359 degrees).
    Much more natural and efficient robot movement!
    """
    
    def __init__(self, map_builder, unit_scale=1.0):
        """
        Initialize smooth path converter.
        
        Args:
            map_builder: MapBuilder instance
            unit_scale: Scale factor for distance
        """
        self.map_builder = map_builder
        self.unit_scale = unit_scale
        
        print(f"Smooth Path Converter initialized (360° angles)")
    
    def convert_path_to_commands(self, path, start_heading=0, max_move_distance=7):
        """
        Convert smoothed path to robot commands with continuous angles.
        Uses IMAGE coordinate system where:
        - 0° = East (right)
        - 90° = South (down)
        - 180° = West (left)
        - 270° = North (up)
        
        OPTIMIZATION: Splits long movements into chunks of max_move_distance
        to allow more frequent path reevaluation.
        
        Args:
            path (list): Path as list of (x, y) tuples
            start_heading (float): Initial robot heading in degrees (0-359)
            max_move_distance (int): Maximum distance per move command (default: 7)
            
        Returns:
            list: Command strings like "turn(37.5)" and "move(7)"
        """
        if not path or len(path) < 2:
            print("⚠ Path too short to convert")
            return []
        
        commands = []
        current_heading = start_heading
        
        # Process each segment
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            # Calculate angle to next waypoint IN IMAGE COORDINATES
            dx = next_point[0] - current[0]
            dy = next_point[1] - current[1]
            
            # Calculate target angle using atan2
            # For IMAGE coordinates: 0° = East, 90° = South
            # Standard atan2(dy, dx) works perfectly for this!
            target_angle_rad = np.arctan2(dy, dx)
            target_angle_deg = np.degrees(target_angle_rad)
            
            # Normalize to 0-360
            if target_angle_deg < 0:
                target_angle_deg += 360
            
            # Round to nearest 0.1 degree
            target_angle_deg = round(target_angle_deg, 1)
            
            # Calculate turn needed
            if abs(target_angle_deg - current_heading) > 0.5:
                commands.append(f"turn({target_angle_deg:.1f})")
                current_heading = target_angle_deg
            
            # Calculate distance
            distance = np.sqrt(dx*dx + dy*dy)
            scaled_distance = distance * self.unit_scale
            
            # OPTIMIZATION: Split long movements into chunks
            if scaled_distance > 0.1:
                # Split into chunks of max_move_distance
                remaining = scaled_distance
                while remaining > 0.1:
                    chunk = min(remaining, max_move_distance)
                    commands.append(f"move({chunk:.1f})")
                    remaining -= chunk
        
        print(f"✓ Converted to {len(commands)} commands (max {max_move_distance} per move)")
        return commands
    
    def optimize_commands(self, commands):
        """
        Optimize commands by combining consecutive moves.
        
        Args:
            commands (list): List of command strings
            
        Returns:
            list: Optimized command list
        """
        if not commands:
            return []
        
        optimized = []
        i = 0
        
        while i < len(commands):
            cmd = commands[i]
            
            if cmd.startswith('move('):
                # Extract distance
                dist = float(cmd[5:-1])
                
                # Look for consecutive moves (same heading)
                j = i + 1
                while j < len(commands) and commands[j].startswith('move('):
                    dist += float(commands[j][5:-1])
                    j += 1
                
                # Add combined move
                optimized.append(f'move({dist:.1f})')
                i = j
            else:
                # It's a turn, add it
                optimized.append(cmd)
                i += 1
        
        if len(commands) != len(optimized):
            print(f"  Optimized: {len(commands)} → {len(optimized)} commands")
        
        return optimized
    
    def save_commands(self, commands, filename="outputs/robot_path.txt"):
        """
        Save commands to file.
        
        Args:
            commands (list): List of command strings
            filename (str): Output filename
            
        Returns:
            bool: True if save successful
        """
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                for cmd in commands:
                    f.write(cmd + '\n')
            
            print(f"✓ Robot commands saved to {filename}")
            print(f"  Total commands: {len(commands)}")
            
            # Print statistics
            turn_count = sum(1 for cmd in commands if cmd.startswith('turn'))
            move_count = sum(1 for cmd in commands if cmd.startswith('move'))
            print(f"  Turns: {turn_count}")
            print(f"  Moves: {move_count}")
            
            # Print angle range
            angles = [float(cmd[5:-1]) for cmd in commands if cmd.startswith('turn')]
            if angles:
                print(f"  Angle range: {min(angles):.1f}° to {max(angles):.1f}°")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to save commands: {str(e)}")
            return False
    
    def preview_commands(self, commands, max_lines=20):
        """
        Print preview of commands.
        
        Args:
            commands (list): List of command strings
            max_lines (int): Maximum lines to print
        """
        print(f"\nCommand Preview (showing first {max_lines}):")
        print("-" * 50)
        
        for i, cmd in enumerate(commands[:max_lines]):
            # Add interpretation
            if cmd.startswith('turn('):
                angle = float(cmd[5:-1])
                direction = self._angle_to_direction_name(angle)
                print(f"  {i+1}. {cmd:20s}  # {direction}")
            elif cmd.startswith('move('):
                dist = float(cmd[5:-1])
                print(f"  {i+1}. {cmd:20s}  # {dist:.1f} units forward")
            else:
                print(f"  {i+1}. {cmd}")
        
        if len(commands) > max_lines:
            print(f"  ... ({len(commands) - max_lines} more commands)")
        print("-" * 50)
    
    def _angle_to_direction_name(self, angle):
        """
        Convert angle to approximate direction name.
        Helps with understanding what the robot is doing.
        """
        # Normalize to 0-360
        angle = angle % 360
        
        # Determine direction
        if 337.5 <= angle or angle < 22.5:
            return "East"
        elif 22.5 <= angle < 67.5:
            return "Northeast"
        elif 67.5 <= angle < 112.5:
            return "North"
        elif 112.5 <= angle < 157.5:
            return "Northwest"
        elif 157.5 <= angle < 202.5:
            return "West"
        elif 202.5 <= angle < 247.5:
            return "Southwest"
        elif 247.5 <= angle < 292.5:
            return "South"
        elif 292.5 <= angle < 337.5:
            return "Southeast"
        else:
            return f"{angle:.1f}°"
    
    def visualize_commands(self, commands, path):
        """
        Create a visualization showing commands along the path.
        """
        import cv2
        
        vis = self.map_builder.visualize(path=path)
        
        # Annotate commands along path
        scale = 5
        cmd_idx = 0
        
        for i, waypoint in enumerate(path[:-1]):
            x, y = waypoint
            px = x * scale
            py = y * scale
            
            # Show command number
            cv2.putText(vis, f"#{cmd_idx}", (px, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            cmd_idx += 1
            if cmd_idx < len(commands) and commands[cmd_idx].startswith('move'):
                cmd_idx += 1
        
        return vis


# Example usage
if __name__ == "__main__":
    print("Testing Smooth Path Converter")
    print("=" * 60)
    
    from map_builder import MapBuilder
    from astar_smooth import AStarSmoothPlanner
    import numpy as np
    
    # Create map
    map_builder = MapBuilder(width=100, height=100)
    grid = np.ones((100, 100), dtype=np.uint8)
    grid[30:50, 40:42] = 0
    map_builder.update_from_segmentation(grid)
    
    # Plan smooth path
    planner = AStarSmoothPlanner(map_builder)
    start = (10, 50)
    goal = (90, 50)
    
    print(f"\nPlanning path from {start} to {goal}...")
    path = planner.plan(start, goal)
    
    if path:
        print(f"✓ Path found: {len(path)} waypoints")
        
        # Convert to commands
        converter = SmoothPathConverter(map_builder, unit_scale=1.0)
        commands = converter.convert_path_to_commands(path, start_heading=0)
        
        # Optimize
        commands = converter.optimize_commands(commands)
        
        # Preview
        converter.preview_commands(commands)
        
        # Save
        converter.save_commands(commands)
        
        print("\n✓ Test complete!")
        print("Check outputs/robot_path.txt for smooth 360° commands!")
    else:
        print("✗ No path found")
