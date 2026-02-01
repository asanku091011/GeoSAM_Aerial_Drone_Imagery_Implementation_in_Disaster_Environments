"""
Enhanced A* with 360-degree smooth paths
Uses hybrid approach: grid-based planning + continuous angle smoothing
"""

import numpy as np
import heapq
import config
import math


class AStarSmoothPlanner:
    """
    Enhanced A* pathfinding with 360-degree smooth paths.
    
    Approach:
    1. Plan on grid using 8-directional movement (fast)
    2. Smooth the path to remove unnecessary waypoints
    3. Calculate continuous angles between waypoints
    
    This gives you smooth, natural-looking paths with any angle!
    """
    
    def __init__(self, map_builder):
        """
        Initialize smooth A* planner.
        
        Args:
            map_builder: MapBuilder instance containing the navigation grid
        """
        self.map_builder = map_builder
        self.allow_diagonal = True
        
        print(f"A* Smooth Planner initialized (360-degree paths)")
    
    def plan(self, start, goal):
        """
        Plan a smooth path from start to goal.
        
        Args:
            start (tuple): Start position (x, y) in grid coordinates
            goal (tuple): Goal position (x, y) in grid coordinates
            
        Returns:
            list: Smoothed path with continuous angles
        """
        # Step 1: Get basic grid path using standard A*
        grid_path = self._plan_grid_path(start, goal)
        
        if not grid_path:
            return None
        
        # Step 2: Smooth the path (remove unnecessary waypoints)
        smoothed_path = self._smooth_path(grid_path)
        
        # Step 3: Optimize further with line-of-sight
        optimized_path = self._optimize_with_line_of_sight(smoothed_path)
        
        print(f"  Path optimization: {len(grid_path)} → {len(smoothed_path)} → {len(optimized_path)} waypoints")
        
        return optimized_path
    
    def _plan_grid_path(self, start, goal):
        """Standard A* on grid (8-directional)."""
        if not self._validate_positions(start, goal):
            return None
        
        # Initialize
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        counter = 1
        explored = set()
        
        # A* main loop
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            if current in explored:
                continue
            
            explored.add(current)
            
            # Check all 8 directions
            neighbors = self.map_builder.get_neighbors(
                current[0], current[1], 
                allow_diagonal=True
            )
            
            for neighbor in neighbors:
                move_cost = self._get_move_cost(current, neighbor)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
        
        print("⚠ A*: No path found")
        return None
    
    def _smooth_path(self, path):
        """
        Remove unnecessary waypoints using Douglas-Peucker algorithm.
        This creates smoother paths with fewer turns.
        
        Args:
            path (list): Original path
            
        Returns:
            list: Smoothed path
        """
        if len(path) <= 2:
            return path
        
        # Use Douglas-Peucker simplification
        epsilon = 1.5  # Tolerance (higher = more aggressive smoothing)
        return self._douglas_peucker(path, epsilon)
    
    def _douglas_peucker(self, points, epsilon):
        """
        Douglas-Peucker line simplification algorithm.
        Removes points that don't significantly change the path shape.
        """
        if len(points) <= 2:
            return points
        
        # Find point with maximum distance from line segment
        dmax = 0
        index = 0
        end = len(points) - 1
        
        for i in range(1, end):
            d = self._perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            rec_results1 = self._douglas_peucker(points[:index+1], epsilon)
            rec_results2 = self._douglas_peucker(points[index:], epsilon)
            
            # Build result list
            result = rec_results1[:-1] + rec_results2
        else:
            result = [points[0], points[end]]
        
        return result
    
    def _perpendicular_distance(self, point, line_start, line_end):
        """
        Calculate perpendicular distance from point to line segment.
        """
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        # If line segment has zero length
        if dx == 0 and dy == 0:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Calculate perpendicular distance
        numerator = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt(dx**2 + dy**2)
        
        return numerator / denominator
    
    def _optimize_with_line_of_sight(self, path):
        """
        Further optimize by removing waypoints where we have line-of-sight.
        This creates the smoothest possible path.
        """
        if len(path) <= 2:
            return path
        
        optimized = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to skip as many points as possible
            for next_idx in range(len(path) - 1, current_idx, -1):
                if self._has_line_of_sight(path[current_idx], path[next_idx]):
                    optimized.append(path[next_idx])
                    current_idx = next_idx
                    break
        
        return optimized
    
    def _has_line_of_sight(self, start, end):
        """
        Check if there's a clear line of sight between two points.
        Uses Bresenham's line algorithm to check all cells along the line.
        """
        x0, y0 = start
        x1, y1 = end
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # Check if current cell is free
            if not self.map_builder.is_cell_free(x, y):
                return False
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def _validate_positions(self, start, goal):
        """Validate start and goal positions."""
        if not self.map_builder.is_valid_cell(start[0], start[1]):
            print(f"✗ A*: Start position {start} is out of bounds")
            return False
        
        if not self.map_builder.is_valid_cell(goal[0], goal[1]):
            print(f"✗ A*: Goal position {goal} is out of bounds")
            return False
        
        if not self.map_builder.is_cell_free(start[0], start[1]):
            print(f"✗ A*: Start position {start} is in obstacle")
            return False
        
        if not self.map_builder.is_cell_free(goal[0], goal[1]):
            print(f"✗ A*: Goal position {goal} is in obstacle")
            return False
        
        return True
    
    def _heuristic(self, a, b):
        """Euclidean distance heuristic."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return math.sqrt(dx*dx + dy*dy)
    
    def _get_move_cost(self, from_pos, to_pos):
        """Calculate movement cost (Euclidean distance)."""
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        return math.sqrt(dx*dx + dy*dy)
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from start to goal."""
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def get_path_angles(self, path):
        """
        Calculate continuous angles for each segment of the path.
        Returns angles in degrees (0-359).
        
        Args:
            path (list): Path as list of (x, y) tuples
            
        Returns:
            list: List of (waypoint, angle_to_next) tuples
        """
        if len(path) < 2:
            return []
        
        angles = []
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Calculate angle using atan2
            dx = x2 - x1
            dy = y2 - y1
            
            # atan2 returns angle in radians, convert to degrees
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            
            # Normalize to 0-360 range
            if angle_deg < 0:
                angle_deg += 360
            
            angles.append((path[i], angle_deg))
        
        # Last point has no angle
        angles.append((path[-1], None))
        
        return angles
    
    def visualize_smooth_path(self, path):
        """
        Create visualization showing the smooth path with angle information.
        """
        import cv2
        
        # Get base visualization
        vis = self.map_builder.visualize(path=path)
        
        # Get angles
        path_angles = self.get_path_angles(path)
        
        # Draw angles on visualization
        scale = 5  # From map_builder
        
        for i, (waypoint, angle) in enumerate(path_angles[:-1]):
            x, y = waypoint
            px = x * scale
            py = y * scale
            
            # Draw angle indicator (small arrow)
            if angle is not None:
                # Calculate arrow end point
                arrow_len = 15
                angle_rad = math.radians(angle)
                end_x = int(px + arrow_len * math.cos(angle_rad))
                end_y = int(py + arrow_len * math.sin(angle_rad))
                
                # Draw arrow
                cv2.arrowedLine(vis, (px, py), (end_x, end_y), 
                              (255, 165, 0), 2, tipLength=0.3)  # Orange arrow
                
                # Draw angle text
                cv2.putText(vis, f"{int(angle)}°", (px + 5, py - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        
        return vis


# Example usage and testing
if __name__ == "__main__":
    print("Testing A* Smooth Planner (360-degree paths)")
    print("=" * 60)
    
    from map_builder import MapBuilder
    import cv2
    import time
    
    # Create test map
    map_builder = MapBuilder(width=100, height=100)
    
    # Create obstacles
    grid = np.ones((100, 100), dtype=np.uint8)
    grid[30:60, 40:42] = 0  # Vertical wall
    grid[30:32, 40:60] = 0  # Horizontal wall
    grid[60:80, 65:67] = 0  # Another vertical wall
    
    map_builder.update_from_segmentation(grid)
    
    # Create planner
    planner = AStarSmoothPlanner(map_builder)
    
    # Plan path
    start = (10, 50)
    goal = (90, 50)
    
    print(f"\nPlanning smooth path from {start} to {goal}...")
    
    t0 = time.time()
    path = planner.plan(start, goal)
    t1 = time.time()
    
    if path:
        print(f"\n✓ Path found!")
        print(f"  Waypoints: {len(path)}")
        print(f"  Planning time: {(t1-t0)*1000:.1f}ms")
        
        # Get angles
        angles = planner.get_path_angles(path)
        
        print(f"\n  Path segments with angles:")
        for i, (waypoint, angle) in enumerate(angles):
            if angle is not None:
                print(f"    {i}. {waypoint} → {angle:.1f}°")
            else:
                print(f"    {i}. {waypoint} (goal)")
        
        # Visualize
        print("\n  Creating visualizations...")
        
        # Standard visualization
        vis1 = map_builder.visualize(path=path, start=start, goal=goal)
        cv2.imshow("1. Smooth Path", vis1)
        
        # With angle indicators
        vis2 = planner.visualize_smooth_path(path)
        cv2.imshow("2. Path with Angles (Orange Arrows)", vis2)
        
        print("\nVisualization windows open!")
        print("  Orange arrows show direction at each waypoint")
        print("  Numbers show exact angle in degrees")
        print("\nPress any key to close...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print("✗ No path found")
    
    print("\n✓ Test complete!")
