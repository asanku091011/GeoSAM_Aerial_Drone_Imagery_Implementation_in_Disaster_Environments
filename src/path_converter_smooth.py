"""
SMOOTH Path Converter with 360° Angles
- Prevents tiny movements (enforces minimum distance)
- Splits large movements into max 7 units
- Optimizes consecutive turns/moves
"""

import numpy as np


class SmoothPathConverter:
    """
    Converts paths to smooth robot commands.
    
    Features:
    - 360° continuous angle support
    - Minimum movement distance (no move(1) spam!)
    - Maximum movement splitting (7 units max)
    - Command optimization
    """
    
    # Movement constraints
    MIN_MOVE_DISTANCE = 5   # Minimum distance to move (prevents move(1))
    MAX_MOVE_DISTANCE = 20   # Maximum distance per command
    
    def __init__(self, map_builder, unit_scale=1.0):
        self.map_builder = map_builder
        self.unit_scale = unit_scale
        print(f"Smooth Path Converter initialized (360° angles)")
        print(f"  Min move: {self.MIN_MOVE_DISTANCE} units")
        print(f"  Max move: {self.MAX_MOVE_DISTANCE} units")
    
    def convert_path_to_commands(self, path, start_heading=0):
        """
        Convert path to smooth turn() and move() commands.
        
        Enforces minimum movement distance to prevent tiny movements.
        Uses continuous angle calculation for smooth turns.
        """
        if not path or len(path) < 2:
            return []
        
        commands = []
        current_heading = start_heading
        i = 0
        
        while i < len(path) - 1:
            # Look ahead to accumulate distance in same direction
            start_idx = i
            total_distance = 0
            current_direction = None
            
            # Scan ahead to find all moves in roughly same direction
            while i < len(path) - 1:
                dx = path[i+1][0] - path[i][0]
                dy = path[i+1][1] - path[i][1]
                
                if dx == 0 and dy == 0:
                    i += 1
                    continue
                
                # Calculate heading for this segment
                segment_heading = self._calculate_smooth_heading(dx, dy)
                
                if current_direction is None:
                    current_direction = segment_heading
                
                # Check if still in same direction (within 5 degrees)
                heading_diff = abs(segment_heading - current_direction)
                if heading_diff > 180:
                    heading_diff = 360 - heading_diff
                
                if heading_diff < 5.0:  # Same direction
                    # Calculate actual Euclidean distance (important for diagonals!)
                    step_distance = np.sqrt(dx**2 + dy**2)
                    total_distance += step_distance
                    i += 1
                else:
                    break  # Direction changed
            
            # Skip if we didn't accumulate enough distance (unless final segment)
            if total_distance < self.MIN_MOVE_DISTANCE and i < len(path) - 1:
                continue
            
            # Add turn command if direction changed
            if current_direction is not None:
                # Calculate shortest angular difference (handle 0°/360° wraparound)
                diff = abs(current_direction - current_heading)
                if diff > 180:
                    diff = 360 - diff
                
                # Only turn if difference > 5° (prevents turn(0.0) spam)
                if diff > 5.0:
                    commands.append(f"turn({current_direction:.1f})")
                    current_heading = current_direction
            
            # Add move commands (split into chunks if needed)
            while total_distance > 0:
                chunk = min(total_distance, self.MAX_MOVE_DISTANCE)
                commands.append(f"move({chunk * self.unit_scale:.1f})")
                total_distance -= chunk
        
        print(f"  Converted to {len(commands)} commands (min {self.MIN_MOVE_DISTANCE}, max {self.MAX_MOVE_DISTANCE} per move)")
        return commands
    
    def _calculate_smooth_heading(self, dx, dy):
        """
        Calculate heading using atan2 for continuous 360° angles.
        
        Returns angle in degrees (0-360).
        """
        if dx == 0 and dy == 0:
            return 0
        
        # Use atan2 for smooth angles
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def optimize_commands(self, commands):
        """
        Optimize command sequence.
        
        - Removes redundant turns (including turn to same angle)
        - Combines consecutive moves (respecting max distance)
        - Preserves movement splitting
        """
        if not commands:
            return []
        
        optimized = []
        i = 0
        
        while i < len(commands):
            cmd = commands[i]
            
            if cmd.startswith('turn('):
                current_angle = float(cmd[5:-1])
                
                # Look ahead to skip redundant turns
                j = i + 1
                while j < len(commands) and commands[j].startswith('turn('):
                    # Keep only the final turn in sequence
                    current_angle = float(commands[j][5:-1])
                    j += 1
                
                # Check if we're turning to the same angle we're already at
                if i > 0:
                    # Find last turn command
                    last_angle = None
                    for prev_cmd in reversed(optimized):
                        if prev_cmd.startswith('turn('):
                            last_angle = float(prev_cmd[5:-1])
                            break
                    
                    # Skip if turning to same angle (handle wraparound)
                    if last_angle is not None:
                        diff = abs(current_angle - last_angle)
                        if diff > 180:
                            diff = 360 - diff
                        
                        if diff < 5.0:  # Within 5° = same angle
                            i = j
                            continue
                
                optimized.append(f"turn({current_angle:.1f})")
                i = j
            
            elif cmd.startswith('move('):
                # Combine consecutive moves (respecting max distance)
                total = float(cmd[5:-1])
                j = i + 1
                
                while j < len(commands) and commands[j].startswith('move('):
                    total += float(commands[j][5:-1])
                    j += 1
                
                # Split if needed
                while total > 0:
                    chunk = min(total, self.MAX_MOVE_DISTANCE)
                    optimized.append(f"move({chunk:.1f})")
                    total -= chunk
                
                i = j
            
            else:
                optimized.append(cmd)
                i += 1
        
        print(f"  Optimized: {len(commands)} → {len(optimized)} commands")
        return optimized
    
    def save_commands(self, commands, filename="outputs/robot_path.txt"):
        """Save commands to file."""
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                for cmd in commands:
                    f.write(cmd + '\n')
            return True
        except:
            return False
    
    def preview_commands(self, commands, max_lines=10):
        """Print command preview."""
        print(f"\nCommand Preview:")
        for i, cmd in enumerate(commands[:max_lines]):
            print(f"  {i+1}. {cmd}")
        if len(commands) > max_lines:
            print(f"  ... +{len(commands)-max_lines} more")