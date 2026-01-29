"""
Map builder module that converts segmentation masks into navigation grids.
Creates and maintains a 2D occupancy grid for path planning algorithms.
"""

import numpy as np
import cv2
import config

class MapBuilder:
    """
    Converts segmentation output into grid-based navigation maps.
    
    The map is a 2D grid where:
    - 0 = obstacle/unsafe (cannot navigate)
    - 1 = free space/safe (can navigate)
    
    This class also handles map updates, obstacle inflation for safety,
    and change detection for dynamic replanning.
    """
    
    def __init__(self, width=None, height=None):
        """
        Initialize the map builder.
        
        Args:
            width (int): Map width in cells (default from config)
            height (int): Map height in cells (default from config)
        """
        self.width = width or config.MAP_WIDTH
        self.height = height or config.MAP_HEIGHT
        
        # Current navigation map (0=obstacle, 1=free)
        self.grid = np.ones((self.height, self.width), dtype=np.uint8)
        
        # Previous map for change detection
        self.previous_grid = self.grid.copy()
        
        # Metadata
        self.resolution = config.GRID_RESOLUTION  # meters per cell
        self.last_update_time = 0
        self.update_count = 0
        
        print(f"Map Builder initialized: {self.width}x{self.height} grid")
        print(f"Resolution: {self.resolution}m per cell")
        print(f"Coverage area: {self.width * self.resolution:.1f}m x {self.height * self.resolution:.1f}m")
    
    def update_from_segmentation(self, segmentation_mask):
        """
        Update the navigation map from a segmentation mask.
        
        Args:
            segmentation_mask (numpy.ndarray): Binary mask (1=safe, 0=unsafe)
            
        Returns:
            bool: True if map was updated successfully
        """
        try:
            # Save previous map for change detection
            self.previous_grid = self.grid.copy()
            
            # Resize segmentation mask to match grid size if needed
            if segmentation_mask.shape != (self.height, self.width):
                mask_resized = cv2.resize(segmentation_mask, 
                                         (self.width, self.height),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = segmentation_mask
            
            # Update grid: 1 where safe, 0 where unsafe
            self.grid = mask_resized.astype(np.uint8)
            
            # Apply obstacle inflation for safety buffer
            if config.OBSTACLE_INFLATION_RADIUS > 0:
                self.grid = self._inflate_obstacles(self.grid)
            
            # Update metadata
            import time
            self.last_update_time = time.time()
            self.update_count += 1
            
            return True
            
        except Exception as e:
            print(f"✗ Map update failed: {str(e)}")
            return False
    
    def _inflate_obstacles(self, grid):
        """
        Inflate obstacles to create a safety buffer.
        This prevents the robot from getting too close to obstacles.
        
        Args:
            grid (numpy.ndarray): Original grid
            
        Returns:
            numpy.ndarray: Grid with inflated obstacles
        """
        # Find all obstacle cells (0s)
        obstacles = (grid == 0).astype(np.uint8)
        
        # Create circular structuring element for inflation
        radius = config.OBSTACLE_INFLATION_RADIUS
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (2*radius + 1, 2*radius + 1))
        
        # Dilate obstacles
        inflated_obstacles = cv2.dilate(obstacles, kernel, iterations=1)
        
        # Convert back: 1 where free, 0 where obstacle
        inflated_grid = (inflated_obstacles == 0).astype(np.uint8)
        
        return inflated_grid
    
    def get_grid(self):
        """
        Get the current navigation grid.
        
        Returns:
            numpy.ndarray: Current grid (0=obstacle, 1=free)
        """
        return self.grid.copy()
    
    def calculate_change_percentage(self):
        """
        Calculate how much the map has changed since last update.
        Used to determine if replanning is needed.
        
        Returns:
            float: Percentage of cells that changed (0-100)
        """
        if self.previous_grid is None:
            return 0.0
        
        # Count cells that changed
        changed_cells = np.sum(self.grid != self.previous_grid)
        total_cells = self.grid.size
        
        change_percentage = (changed_cells / total_cells) * 100
        
        return change_percentage
    
    def is_cell_free(self, x, y):
        """
        Check if a specific cell is free (navigable).
        
        Args:
            x (int): Cell x-coordinate
            y (int): Cell y-coordinate
            
        Returns:
            bool: True if cell is free, False if obstacle or out of bounds
        """
        if not self.is_valid_cell(x, y):
            return False
        
        return self.grid[y, x] == 1
    
    def is_valid_cell(self, x, y):
        """
        Check if cell coordinates are within map bounds.
        
        Args:
            x (int): Cell x-coordinate
            y (int): Cell y-coordinate
            
        Returns:
            bool: True if coordinates are valid
        """
        return 0 <= x < self.width and 0 <= y < self.height
    
    def world_to_grid(self, world_x, world_y):
        """
        Convert world coordinates (meters) to grid coordinates (cells).
        
        Args:
            world_x (float): X position in meters
            world_y (float): Y position in meters
            
        Returns:
            tuple: (grid_x, grid_y) in cells
        """
        grid_x = int(world_x / self.resolution)
        grid_y = int(world_y / self.resolution)
        
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid coordinates (cells) to world coordinates (meters).
        
        Args:
            grid_x (int): X position in cells
            grid_y (int): Y position in cells
            
        Returns:
            tuple: (world_x, world_y) in meters
        """
        world_x = (grid_x + 0.5) * self.resolution  # +0.5 for cell center
        world_y = (grid_y + 0.5) * self.resolution
        
        return (world_x, world_y)
    
    def get_neighbors(self, x, y, allow_diagonal=True):
        """
        Get valid neighboring cells for path planning.
        
        Args:
            x (int): Cell x-coordinate
            y (int): Cell y-coordinate
            allow_diagonal (bool): Include diagonal neighbors
            
        Returns:
            list: List of (x, y) tuples for valid free neighbors
        """
        neighbors = []
        
        # 4-connected neighbors (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Add diagonal moves if allowed
        if allow_diagonal:
            moves.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            
            if self.is_cell_free(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def visualize(self, path=None, start=None, goal=None):
        """
        Create a visualization of the map.
        
        Args:
            path (list): Optional path to draw as (x, y) tuples
            start (tuple): Optional start position (x, y)
            goal (tuple): Optional goal position (x, y)
            
        Returns:
            numpy.ndarray: RGB image of the map
        """
        # Create RGB image: white=free, black=obstacle
        vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        vis[self.grid == 1] = [255, 255, 255]  # Free space = white
        vis[self.grid == 0] = [0, 0, 0]  # Obstacles = black
        
        # Scale up for better visibility
        scale = 5
        vis = cv2.resize(vis, (self.width * scale, self.height * scale), 
                        interpolation=cv2.INTER_NEAREST)
        
        # Draw path if provided
        if path is not None and len(path) > 0:
            for i in range(len(path) - 1):
                pt1 = (int(path[i][0] * scale), int(path[i][1] * scale))
                pt2 = (int(path[i+1][0] * scale), int(path[i+1][1] * scale))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)  # Green path
        
        # Draw start position
        if start is not None:
            center = (int(start[0] * scale), int(start[1] * scale))
            cv2.circle(vis, center, scale * 2, (255, 0, 0), -1)  # Blue start
            cv2.circle(vis, center, scale * 2, (0, 0, 0), 1)  # Black outline
        
        # Draw goal position
        if goal is not None:
            center = (int(goal[0] * scale), int(goal[1] * scale))
            cv2.circle(vis, center, scale * 2, (0, 0, 255), -1)  # Red goal
            cv2.circle(vis, center, scale * 2, (0, 0, 0), 1)  # Black outline
        
        return vis
    
    def get_statistics(self):
        """
        Get statistics about the current map.
        
        Returns:
            dict: Map statistics
        """
        free_cells = np.sum(self.grid == 1)
        obstacle_cells = np.sum(self.grid == 0)
        total_cells = self.grid.size
        
        return {
            'width': self.width,
            'height': self.height,
            'resolution': self.resolution,
            'free_cells': free_cells,
            'obstacle_cells': obstacle_cells,
            'free_percentage': (free_cells / total_cells) * 100,
            'obstacle_percentage': (obstacle_cells / total_cells) * 100,
            'update_count': self.update_count,
            'last_change_percentage': self.calculate_change_percentage()
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Map Builder Module")
    print("=" * 50)
    
    # Create map builder
    map_builder = MapBuilder()
    
    # Create synthetic segmentation mask
    mask = np.ones((config.MAP_HEIGHT, config.MAP_WIDTH), dtype=np.uint8)
    
    # Add some obstacles
    mask[20:40, 20:40] = 0  # Square obstacle
    mask[60:80, 60:80] = 0  # Another obstacle
    cv2.circle(mask, (50, 50), 15, 0, -1)  # Circular obstacle
    
    print("\nUpdating map from segmentation...")
    success = map_builder.update_from_segmentation(mask)
    
    if success:
        # Get statistics
        stats = map_builder.get_statistics()
        print(f"\n✓ Map updated successfully!")
        print(f"  Free space: {stats['free_percentage']:.1f}%")
        print(f"  Obstacles: {stats['obstacle_percentage']:.1f}%")
        
        # Test pathfinding helpers
        print(f"\nTesting navigation helpers...")
        print(f"  Cell (30, 30) is free: {map_builder.is_cell_free(30, 30)}")
        print(f"  Cell (25, 25) is free: {map_builder.is_cell_free(25, 25)}")
        
        neighbors = map_builder.get_neighbors(10, 10)
        print(f"  Cell (10, 10) has {len(neighbors)} free neighbors")
        
        # Visualize
        vis = map_builder.visualize(start=(10, 10), goal=(90, 90))
        cv2.imshow("Navigation Map", vis)
        
        print("\nPress any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print("✗ Map update failed")