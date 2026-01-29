"""
Path Converter - Converts planned paths to robot_path.txt format
Generates turn(angle) and move(distance) commands for Raspberry Pi robot

UPDATED: Now supports 8 directions (including diagonals)
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
    
    Where angle is in degrees (0, 45, 90, 135, 180, 225, 270, 315) 
    and distance is in units.
    
    UPDATED: Now supports diagonal movements!
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
        
        # 8-direction mapping: angle in degrees
        self.NORTH = 0       # Up
        self.NORTHEAST = 45  # Up-Right
        self.EAST = 90       # Right
        self.SOUTHEAST = 135 # Down-Right
        self.SOUTH = 180     # Down
        self.SOUTHWEST = 225 # Down-Left
        self.WEST = 270      # Left
        self.NORTHWEST = 315 # Up-Left
        
        print(f"Path Converter initialized (unit_scale: {unit_scale})")
        print(f"  Supports 8 directions including diagonals")
    
    def convert_path_to_commands(self, path, start_heading=0):
        """
        Convert a path (list of grid waypoints) to turn/move commands.
        
        Args:
            path (list): Path as list of (x, y) tuples in grid coordinates
            start_heading (int): Initial robot heading in degrees (0, 45, 90, etc.)
            
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
        NOW SUPPORTS 8 DIRECTIONS (including diagonals)!
        
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
        
        # Determine initial direction (8 directions)
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]
        
        if dx == 0 and dy == 0:
            return None, 0
        
        # Classify into 8 directions based on dx and dy
        direction = self._classify_direction(dx, dy)
        
        # Count consecutive cells in same direction
        distance = 1
        idx = start_idx + 1
        
        while idx < len(path) - 1:
            curr = path[idx]
            nxt = path[idx + 1]
            
            new_dx = nxt[0] - curr[0]
            new_dy = nxt[1] - curr[1]
            
            # Check if still moving in same direction
            new_direction = self._classify_direction(new_dx, new_dy)
            
            if new_direction != direction:
                break  # Direction changed
            
            distance += 1
            idx += 1
        
        return direction, distance
    
    def _classify_direction(self, dx, dy):
        """
        Classify a movement vector into one of 8 directions.
        
        Args:
            dx (int): Change in x
            dy (int): Change in y
            
        Returns:
            str: Direction name (e.g., 'north', 'northeast', 'east', etc.)
        """
        # Normalize to -1, 0, or 1
        norm_dx = 0 if dx == 0 else (1 if dx > 0 else -1)
        norm_dy = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        # Map to 8 directions
        direction_map = {
            (0, -1): 'north',      # Up
            (1, -1): 'northeast',  # Up-Right
            (1, 0): 'east',        # Right
            (1, 1): 'southeast',   # Down-Right
            (0, 1): 'south',       # Down
            (-1, 1): 'southwest',  # Down-Left
            (-1, 0): 'west',       # Left
            (-1, -1): 'northwest'  # Up-Left
        }
        
        return direction_map.get((norm_dx, norm_dy), None)
    
    def _direction_to_heading(self, direction):
        """
        Convert direction string to heading in degrees.
        NOW SUPPORTS 8 DIRECTIONS!
        
        Args:
            direction (str): Direction name
            
        Returns:
            int: Heading in degrees (0, 45, 90, 135, 180, 225, 270, 315)
        """
        direction_map = {
            'north': self.NORTH,          # 0
            'northeast': self.NORTHEAST,  # 45
            'east': self.EAST,            # 90
            'southeast': self.SOUTHEAST,  # 135
            'south': self.SOUTH,          # 180
            'southwest': self.SOUTHWEST,  # 225
            'west': self.WEST,            # 270
            'northwest': self.NORTHWEST   # 315
        }
        return direction_map.get(direction, self.NORTH)
    
    def _calculate_turn(self, current_heading, target_heading):
        """
        Calculate the turn angle from current to target heading.
        Always returns the target heading (absolute, not relative).
        
        Args:
            current_heading (int): Current heading in degrees
            target_heading (int): Target heading in degrees
            
        Returns:
            int: Target heading (0, 45, 90, 135, 180, 225, 270, or 315)
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
            
            # Print unique turn angles used
            turn_angles = set()
            for cmd in commands:
                if cmd.startswith('turn('):
                    angle = int(cmd[5:-1])
                    turn_angles.add(angle)
            
            if turn_angles:
                print(f"  Turn angles used: {sorted(turn_angles)}°")
            
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
    print("Testing Path Converter (8-Direction Support)")
    print("=" * 50)
    
    from map_builder import MapBuilder
    from astar import AStarPlanner
    import numpy as np
    
    # Create map
    map_builder = MapBuilder(width=50, height=50)
    grid = np.ones((50, 50), dtype=np.uint8)
    grid[15:35, 20:22] = 0  # Add an obstacle
    map_builder.update_from_segmentation(grid)
    
    # Plan a diagonal path (this will test diagonal support!)
    planner = AStarPlanner(map_builder)
    start = (5, 5)
    goal = (45, 45)  # Diagonal path!
    
    print(f"\nPlanning DIAGONAL path from {start} to {goal}...")
    path = planner.plan(start, goal)
    
    if path:
        print(f"✓ Path found: {len(path)} waypoints")
        
        # Check if path is diagonal
        if len(path) >= 2:
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
            if abs(dx) == abs(dy) and dx != 0:
                print(f"  ✓ Path is diagonal! (dx={dx}, dy={dy})")
            else:
                print(f"  Path direction: dx={dx}, dy={dy}")
        
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
        
        # Verify diagonal turns
        for cmd in commands[:5]:  # Check first 5 commands
            if 'turn(135)' in cmd or 'turn(45)' in cmd or 'turn(315)' in cmd or 'turn(225)' in cmd:
                print(f"\n🎉 SUCCESS: Found diagonal turn angle: {cmd}")
                break
    else:
        print("✗ No path found")