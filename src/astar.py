"""
A* path planning algorithm for grid-based navigation.
A* is optimal and guarantees the shortest path if one exists.
"""

import numpy as np
import heapq
import config

class AStarPlanner:
    """
    A* pathfinding algorithm implementation.
    
    A* uses a heuristic (estimated distance to goal) to efficiently
    search for the optimal path. It's guaranteed to find the shortest
    path if one exists.
    
    The algorithm maintains:
    - g_score: actual cost from start to current node
    - h_score: heuristic estimate from current to goal
    - f_score: g_score + h_score (total estimated cost)
    """
    
    def __init__(self, map_builder):
        """
        Initialize A* planner.
        
        Args:
            map_builder: MapBuilder instance containing the navigation grid
        """
        self.map_builder = map_builder
        self.allow_diagonal = config.ASTAR_DIAGONAL_ALLOWED
        
        print(f"A* Planner initialized (diagonal movement: {self.allow_diagonal})")
    
    def plan(self, start, goal):
        """
        Plan a path from start to goal using A*.
        
        Args:
            start (tuple): Start position (x, y) in grid coordinates
            goal (tuple): Goal position (x, y) in grid coordinates
            
        Returns:
            list: Path as list of (x, y) tuples, or None if no path exists
        """
        # Validate inputs
        if not self._validate_positions(start, goal):
            return None
        
        # Initialize data structures
        open_set = []  # Priority queue: (f_score, counter, node)
        heapq.heappush(open_set, (0, 0, start))
        
        came_from = {}  # For reconstructing path
        g_score = {start: 0}  # Cost from start to each node
        f_score = {start: self._heuristic(start, goal)}  # Estimated total cost
        
        counter = 1  # Tie-breaker for heap
        explored = set()  # Closed set
        
        # A* main loop
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # Goal reached!
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return path
            
            # Skip if already explored
            if current in explored:
                continue
            
            explored.add(current)
            
            # Check all neighbors
            neighbors = self.map_builder.get_neighbors(
                current[0], current[1], 
                allow_diagonal=self.allow_diagonal
            )
            
            for neighbor in neighbors:
                # Calculate tentative g_score
                move_cost = self._get_move_cost(current, neighbor)
                tentative_g = g_score[current] + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Record this path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    
                    # Add to open set
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
        
        # No path found
        print("⚠ A*: No path found to goal")
        return None
    
    def _validate_positions(self, start, goal):
        """
        Validate that start and goal positions are valid.
        
        Args:
            start (tuple): Start position
            goal (tuple): Goal position
            
        Returns:
            bool: True if positions are valid
        """
        # Check if positions are within bounds
        if not self.map_builder.is_valid_cell(start[0], start[1]):
            print(f"✗ A*: Start position {start} is out of bounds")
            return False
        
        if not self.map_builder.is_valid_cell(goal[0], goal[1]):
            print(f"✗ A*: Goal position {goal} is out of bounds")
            return False
        
        # Check if positions are free (not in obstacle)
        if not self.map_builder.is_cell_free(start[0], start[1]):
            print(f"✗ A*: Start position {start} is in obstacle")
            return False
        
        if not self.map_builder.is_cell_free(goal[0], goal[1]):
            print(f"✗ A*: Goal position {goal} is in obstacle")
            return False
        
        return True
    
    def _heuristic(self, a, b):
        """
        Calculate heuristic distance between two points.
        Uses Euclidean distance for better accuracy.
        
        Args:
            a (tuple): First position (x, y)
            b (tuple): Second position (x, y)
            
        Returns:
            float: Estimated distance
        """
        # Euclidean distance
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        
        if self.allow_diagonal:
            # Octile distance (better heuristic for diagonal movement)
            return config.ASTAR_DIAGONAL_COST * min(dx, dy) + abs(dx - dy)
        else:
            # Manhattan distance (for 4-connected grid)
            return dx + dy
    
    def _get_move_cost(self, from_pos, to_pos):
        """
        Calculate the cost of moving from one position to another.
        
        Args:
            from_pos (tuple): Starting position
            to_pos (tuple): Ending position
            
        Returns:
            float: Movement cost
        """
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        
        # Diagonal move
        if dx + dy == 2:
            return config.ASTAR_DIAGONAL_COST
        # Straight move
        else:
            return config.ASTAR_STRAIGHT_COST
    
    def _reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from start to goal.
        
        Args:
            came_from (dict): Dictionary mapping each node to its predecessor
            current (tuple): Goal position
            
        Returns:
            list: Path from start to goal as list of (x, y) tuples
        """
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        # Reverse to get start -> goal order
        path.reverse()
        
        return path
    
    def smooth_path(self, path):
        """
        Smooth the path by removing unnecessary waypoints.
        Uses line-of-sight checking to skip intermediate points.
        
        Args:
            path (list): Original path
            
        Returns:
            list: Smoothed path with fewer waypoints
        """
        if not path or len(path) <= 2:
            return path
        
        smoothed = [path[0]]  # Start with first point
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to skip as many points as possible
            for next_idx in range(len(path) - 1, current_idx, -1):
                if self._has_line_of_sight(path[current_idx], path[next_idx]):
                    smoothed.append(path[next_idx])
                    current_idx = next_idx
                    break
        
        return smoothed
    
    def _has_line_of_sight(self, start, end):
        """
        Check if there's a clear line of sight between two points.
        Uses Bresenham's line algorithm.
        
        Args:
            start (tuple): Start position
            end (tuple): End position
            
        Returns:
            bool: True if line of sight is clear
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

# Example usage and testing
if __name__ == "__main__":
    print("Testing A* Path Planning Module")
    print("=" * 50)
    
    # Create a simple test map
    from map_builder import MapBuilder
    
    map_builder = MapBuilder(width=50, height=50)
    
    # Create obstacles
    grid = np.ones((50, 50), dtype=np.uint8)
    grid[15:35, 20:22] = 0  # Vertical wall with gap
    grid[15:17, 20:40] = 0  # Horizontal wall
    
    map_builder.update_from_segmentation(grid)
    
    # Create A* planner
    planner = AStarPlanner(map_builder)
    
    # Plan path
    start = (5, 25)
    goal = (45, 25)
    
    print(f"\nPlanning path from {start} to {goal}...")
    
    import time
    t0 = time.time()
    path = planner.plan(start, goal)
    t1 = time.time()
    
    if path:
        print(f"✓ Path found with {len(path)} waypoints")
        print(f"  Planning time: {(t1-t0)*1000:.1f}ms")
        
        # Smooth path
        smoothed = planner.smooth_path(path)
        print(f"  Smoothed to {len(smoothed)} waypoints")
        
        # Visualize
        import cv2
        vis = map_builder.visualize(path=path, start=start, goal=goal)
        cv2.imshow("A* Path", vis)
        
        vis_smooth = map_builder.visualize(path=smoothed, start=start, goal=goal)
        cv2.imshow("A* Path (Smoothed)", vis_smooth)
        
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("✗ No path found")