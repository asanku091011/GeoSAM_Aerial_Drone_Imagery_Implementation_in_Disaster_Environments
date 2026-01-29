"""
Configuration file for the Disaster Navigation System.
Contains all system parameters, thresholds, and settings.
"""

import os

# =====================================================
# DRONE CONFIGURATION
# =====================================================
DRONE_ENABLED = False  # Set to False to use recorded video for testing
DRONE_VIDEO_PORT = 11111  # UDP port for Tello video stream
DRONE_FRAME_WIDTH = 960
DRONE_FRAME_HEIGHT = 720
DRONE_FPS = 30

# =====================================================
# SEGMENTATION CONFIGURATION
# =====================================================
# Path to the Geo-SAM model weights (you must download these)
GEOSAM_CHECKPOINT = r"C:\Users\sanku\Documents\RSEF\models\sam_vit_h_4b8939.pth"
GEOSAM_MODEL_PATH = "models/geosam_weights.pth"

# Image processing settings
SEGMENTATION_INPUT_SIZE = (512, 512)  # Resize input for faster processing
SAFE_CLASS_LABEL = 1  # Label for navigable terrain
UNSAFE_CLASS_LABEL = 0  # Label for obstacles/unsafe areas

# Confidence threshold for segmentation (0.0 to 1.0)
SEGMENTATION_CONFIDENCE_THRESHOLD = 0.6

# =====================================================
# MAP BUILDING CONFIGURATION
# =====================================================
# Grid resolution in meters (real-world distance per grid cell)
GRID_RESOLUTION = 0.1  # 10 cm per cell

# Map dimensions (cells)
MAP_WIDTH = 100
MAP_HEIGHT = 100

# Obstacle inflation radius (cells) - adds safety buffer around obstacles
# SET TO 0 to avoid false collision detection
OBSTACLE_INFLATION_RADIUS = 0

# Map update settings
MAP_UPDATE_THRESHOLD = 0.15  # 15% change triggers replanning

# =====================================================
# PATH PLANNING CONFIGURATION
# =====================================================
# Available planning algorithms
PLANNING_ALGORITHMS = ['astar', 'rrt_star', 'greedy']
DEFAULT_ALGORITHM = 'astar'

# A* configuration
ASTAR_DIAGONAL_ALLOWED = True
ASTAR_DIAGONAL_COST = 1.414  # sqrt(2)
ASTAR_STRAIGHT_COST = 1.0

# RRT* configuration
RRT_MAX_ITERATIONS = 1000
RRT_STEP_SIZE = 5  # cells
RRT_GOAL_SAMPLE_RATE = 0.1  # 10% chance to sample goal
RRT_SEARCH_RADIUS = 10  # cells for neighbor search

# Greedy configuration
GREEDY_MAX_ITERATIONS = 500
GREEDY_STEP_SIZE = 1  # cells

# =====================================================
# DYNAMIC REPLANNING CONFIGURATION
# =====================================================
# Enable dynamic replanning when environment changes
# SET TO FALSE for simpler navigation
DYNAMIC_REPLANNING_ENABLED = False

# Minimum time between replanning attempts (seconds)
MIN_REPLAN_INTERVAL = 2.0

# Collision detection settings
COLLISION_CHECK_AHEAD = 5  # Check N waypoints ahead for obstacles
COLLISION_TOLERANCE = 1  # cells

# =====================================================
# ROBOT CONTROLLER CONFIGURATION
# =====================================================
# Robot physical parameters
ROBOT_WHEEL_BASE = 0.15  # meters (distance between wheels)
ROBOT_WHEEL_DIAMETER = 0.065  # meters
ROBOT_MAX_LINEAR_SPEED = 0.3  # m/s
ROBOT_MAX_ANGULAR_SPEED = 1.0  # rad/s

# Control parameters
WAYPOINT_REACHED_THRESHOLD = 0.15  # meters
CONTROLLER_UPDATE_RATE = 10  # Hz

# Movement command output
COMMAND_OUTPUT_FILE = "outputs/robot_path.txt"
COMMAND_FORMAT = "json"  # 'json' or 'simple'

# =====================================================
# DATA LOGGING CONFIGURATION
# =====================================================
# Enable logging
LOGGING_ENABLED = True

# Log file paths
LOG_DIR = "logs"
NAVIGATION_LOG_FILE = os.path.join(LOG_DIR, "navigation_log.csv")
EVENT_LOG_FILE = os.path.join(LOG_DIR, "event_log.txt")

# What to log
LOG_PATH_LENGTH = True
LOG_EXECUTION_TIME = True
LOG_COLLISIONS = True
LOG_REPLANNING_EVENTS = True
LOG_SEGMENTATION_TIME = True

# =====================================================
# SYSTEM CONFIGURATION
# =====================================================
# Main loop update rate (Hz)
SYSTEM_UPDATE_RATE = 5

# Safety settings
EMERGENCY_STOP_ENABLED = True
MAX_EXECUTION_TIME = 300  # seconds (5 minutes)

# Visualization settings
ENABLE_VISUALIZATION = True
VISUALIZATION_WINDOW_NAME = "Disaster Navigation System"
SAVE_VISUALIZATION = False  # Set True to save frames (creates many files!)
VISUALIZATION_OUTPUT_DIR = "outputs/visualizations"

# =====================================================
# DIRECTORY STRUCTURE
# =====================================================
# Ensure all necessary directories exist
REQUIRED_DIRS = [
    "models",
    "logs",
    "outputs",
    "outputs/visualizations",
    "data/test_images",
    "data/recorded_videos"  # For testing without drone
]

def create_directories():
    """Create all required directories if they don't exist."""
    for directory in REQUIRED_DIRS:
        os.makedirs(directory, exist_ok=True)
    print("✓ All required directories created/verified")

# =====================================================
# RASPBERRY PI 5 OPTIMIZATION
# =====================================================
# These settings optimize performance on Raspberry Pi 5
USE_THREADING = True  # Enable multi-threading for better performance
NUM_THREADS = 4  # Raspberry Pi 5 has 4 cores

# Memory management
MAX_FRAME_BUFFER_SIZE = 10  # Keep only recent frames in memory
CLEAR_MEMORY_INTERVAL = 50  # Clear unused objects every N iterations

# Power management (important for battery-powered robots)
ENABLE_POWER_SAVING = False  # Set True to reduce CPU usage when idle

# =====================================================
# DEBUG AND TESTING
# =====================================================
DEBUG_MODE = False  # Enable verbose logging
TEST_MODE = False  # Use synthetic data instead of real drone

# Test data settings
TEST_VIDEO_PATH = "data/recorded_videos/test_disaster_scene.mp4"
TEST_START_POSITION = (10, 10)  # Grid coordinates
TEST_GOAL_POSITION = (90, 90)  # Grid coordinates

if __name__ == "__main__":
    # Run this file to create directories and verify configuration
    print("Disaster Navigation System - Configuration")
    print("=" * 50)
    create_directories()
    print(f"✓ Drone enabled: {DRONE_ENABLED}")
    print(f"✓ Default algorithm: {DEFAULT_ALGORITHM}")
    print(f"✓ Grid resolution: {GRID_RESOLUTION}m per cell")
    print(f"✓ Map size: {MAP_WIDTH}x{MAP_HEIGHT} cells")
    print(f"✓ System update rate: {SYSTEM_UPDATE_RATE} Hz")
    print(f"✓ Dynamic replanning: {DYNAMIC_REPLANNING_ENABLED}")
    print(f"✓ Obstacle inflation: {OBSTACLE_INFLATION_RADIUS} cells")
    print("=" * 50)
    print("Configuration loaded successfully!")
