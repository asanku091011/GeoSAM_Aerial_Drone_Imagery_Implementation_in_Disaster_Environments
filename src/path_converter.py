"""
Path Converter - Converts planned paths to robot_path.txt format
Generates turn(angle) and move(distance) commands for Raspberry Pi robot
"""

import numpy as np
import config

class PathConverter:
    """
    Converts a planned path (list of waypoints) into discrete
    turn/move commands that the Raspberry Pi robot can execute.
    
    Output format:
        turn(angle)
        move(distance)
    
    Where angle is in degrees (0, 90, 180, 270) and distance is in units.
    """
    
    def __init__(self, map_builder, unit_scale=1.0):
        """
        Initialize path converter.
        
        Args:
            map_builder: MapBuilder instance for coordinate conversion
            unit_scale: Scale factor for distance (1.0 = grid cells, 10.0 = cm, etc.)
        """
        self.map_builder = map_builder
        self.unit_scale = unit_scale
        
        # Direction mapping: angle in degrees
        self.NORTH = 0
        self.EAST = 90
        self.SOUTH = 180
        self.WEST = 270
        
        print(f"Path Converter initialized (unit_scale: {unit_scale})")
    
    def convert_path_to_commands(self, path, start_heading=0):
        """
        Convert a path (list of grid waypoints) to turn/move commands.
        
        Args:
            path (list): Path as list of (x, y) tuples in grid coordinates
            start_heading (int): Initial robot heading in degrees (0, 90, 180, 270)
            
        Returns:
            list: List of command strings like "turn(90)" and "move(10)"
        """
        if not path or len(path) < 2:
            print("⚠ Path too short to convert")
            return []
        
        commands = []
        current_heading = start_heading
        
        # Process path waypoint by waypoint
        i = 0
        while i < len(path) - 1:
            current = path[i]
            
            # Look ahead and count consecutive moves in same direction
            direction, distance = self._get_direction_and_distance(path, i)
            
            if direction is None:
                i += 1
                continue
            
            # Calculate required turn
            required_heading = self._direction_to_heading(direction)
            
            if required_heading != current_heading:
                # Need to turn
                turn_angle = self._calculate_turn(current_heading, required_heading)
                commands.append(f"turn({turn_angle})")
                current_heading = required_heading
            
            # Move forward
            scaled_distance = int(distance * self.unit_scale)
            if scaled_distance > 0:
                commands.append(f"move({scaled_distance})")
            
            # Skip ahead by the distance we just moved
            i += distance
        
        print(f"✓ Converted path to {len(commands)} commands")
        return commands
    
    def _get_direction_and_distance(self, path, start_idx):
        """
        Get the direction and count consecutive moves in that direction.
        
        Args:
            path (list): Complete path
            start_idx (int): Starting index
            
        Returns:
            tuple: (direction_string, distance_in_cells) or (None, 0)
        """
        if start_idx >= len(path) - 1:
            return None, 0
        
        current = path[start_idx]
        next_point = path[start_idx + 1]
        
        # Determine initial direction
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]
        
        if dx == 0 and dy == 0:
            return None, 0
        
        # Normalize to get direction
        if abs(dx) > abs(dy):
            direction = 'east' if dx > 0 else 'west'
            primary_axis = 0  # x-axis
        else:
            direction = 'south' if dy > 0 else 'north'
            primary_axis = 1  # y-axis
        
        # Count consecutive cells in same direction
        distance = 1
        idx = start_idx + 1
        
        while idx < len(path) - 1:
            curr = path[idx]
            nxt = path[idx + 1]
            
            new_dx = nxt[0] - curr[0]
            new_dy = nxt[1] - curr[1]
            
            # Check if still moving in same direction
            if primary_axis == 0:  # Moving horizontally
                if abs(new_dx) <= abs(new_dy):  # Changed to vertical
                    break
                if (dx > 0) != (new_dx > 0):  # Changed direction
                    break
            else:  # Moving vertically
                if abs(new_dy) <= abs(new_dx):  # Changed to horizontal
                    break
                if (dy > 0) != (new_dy > 0):  # Changed direction
                    break
            
            distance += 1
            idx += 1
        
        return direction, distance
    
    def _direction_to_heading(self, direction):
        """
        Convert direction string to heading in degrees.
        
        Args:
            direction (str): 'north', 'east', 'south', or 'west'
            
        Returns:
            int: Heading in degrees (0, 90, 180, 270)
        """
        direction_map = {
            'north': self.NORTH,    # 0 degrees
            'east': self.EAST,      # 90 degrees
            'south': self.SOUTH,    # 180 degrees
            'west': self.WEST       # 270 degrees
        }
        return direction_map.get(direction, self.NORTH)
    
    def _calculate_turn(self, current_heading, target_heading):
        """
        Calculate the turn angle from current to target heading.
        Always returns the angle to turn to (not the delta).
        
        Args:
            current_heading (int): Current heading in degrees
            target_heading (int): Target heading in degrees
            
        Returns:
            int: Target heading (0, 90, 180, or 270)
        """
        # Your robot expects absolute headings, not relative turns
        return target_heading
    
    def save_commands(self, commands, filename="outputs/robot_path.txt"):
        """
        Save commands to file in the exact format your Pi expects.
        
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
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to save commands: {str(e)}")
            return False
    
    def optimize_commands(self, commands):
        """
        Optimize command list by combining consecutive moves in same direction.
        
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
                dist = int(cmd[5:-1])
                
                # Look ahead for more moves (without turns between)
                j = i + 1
                while j < len(commands) and commands[j].startswith('move('):
                    dist += int(commands[j][5:-1])
                    j += 1
                
                # Add combined move
                optimized.append(f'move({dist})')
                i = j
            else:
                # It's a turn, just add it
                optimized.append(cmd)
                i += 1
        
        print(f"  Optimized: {len(commands)} → {len(optimized)} commands")
        return optimized
    
    def preview_commands(self, commands, max_lines=20):
        """
        Print preview of commands.
        
        Args:
            commands (list): List of command strings
            max_lines (int): Maximum lines to print
        """
        print(f"\nCommand Preview (showing first {max_lines}):")
        print("-" * 40)
        for i, cmd in enumerate(commands[:max_lines]):
            print(f"  {cmd}")
        
        if len(commands) > max_lines:
            print(f"  ... ({len(commands) - max_lines} more commands)")
        print("-" * 40)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Path Converter")
    print("=" * 50)
    
    from map_builder import MapBuilder
    from astar import AStarPlanner
    import numpy as np
    
    # Create map
    map_builder = MapBuilder(width=50, height=50)
    grid = np.ones((50, 50), dtype=np.uint8)
    grid[15:35, 20:22] = 0  # Add an obstacle
    map_builder.update_from_segmentation(grid)
    
    # Plan a path
    planner = AStarPlanner(map_builder)
    start = (5, 25)
    goal = (45, 25)
    
    print(f"\nPlanning path from {start} to {goal}...")
    path = planner.plan(start, goal)
    
    if path:
        print(f"✓ Path found: {len(path)} waypoints")
        
        # Convert to commands
        converter = PathConverter(map_builder, unit_scale=1.0)
        commands = converter.convert_path_to_commands(path, start_heading=0)
        
        # Optimize
        commands = converter.optimize_commands(commands)
        
        # Preview
        converter.preview_commands(commands)
        
        # Save
        converter.save_commands(commands)
        
        print("\n✓ Test complete!")
        print(f"Check outputs/robot_path.txt")
    else:
        print("✗ No path found")