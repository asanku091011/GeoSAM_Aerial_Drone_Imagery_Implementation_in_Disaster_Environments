"""
Drone input module for capturing overhead images from DJI Tello drone.
Windows-compatible version using OpenCV instead of PyAV.
Provides continuous video stream for the navigation system.
"""

import cv2
import numpy as np
import time
import socket
from threading import Thread, Lock
import config

class DroneInput:
    """
    Manages connection to DJI Tello drone and provides continuous video stream.
    
    This class handles:
    - Drone connection and initialization (using raw UDP, no PyAV dependency)
    - Video stream setup and management (using OpenCV)
    - Frame capture and buffering
    - Error handling and reconnection
    
    WINDOWS COMPATIBLE: Uses OpenCV for video instead of PyAV/djitellopy
    """
    
    def __init__(self, use_drone=True):
        """
        Initialize drone connection or video file.
        
        Args:
            use_drone (bool): True to use real drone, False for testing with video file
        """
        self.use_drone = use_drone and config.DRONE_ENABLED
        self.video_capture = None
        self.current_frame = None
        self.frame_lock = Lock()  # Thread-safe frame access
        self.is_streaming = False
        self.capture_thread = None
        
        # For real drone (raw UDP, no djitellopy)
        self.command_socket = None
        self.is_drone_connected = False
        
        # Performance metrics
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        print(f"Initializing drone input (Mode: {'Real Drone' if self.use_drone else 'Test Video'})...")
        
    def connect(self):
        """
        Connect to the drone or open test video file.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.use_drone:
                return self._connect_real_drone()
            else:
                return self._connect_test_video()
        except Exception as e:
            print(f"✗ Connection failed: {str(e)}")
            return False
    
    def _connect_real_drone(self):
        """Connect to real DJI Tello drone using raw UDP (Windows compatible)."""
        try:
            print("Connecting to Tello drone...")
            
            # Create command socket
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.command_socket.bind(('', 9000))
            self.command_socket.settimeout(10.0)
            
            # Send command mode to enter SDK mode
            print("  Entering SDK mode...")
            self._send_command('command')
            
            print(f"✓ Drone connected!")
            self.is_drone_connected = True
            
            # Get battery level
            try:
                battery_response = self._send_command('battery?')
                if battery_response:
                    print(f"  Battery: {battery_response}%")
            except:
                print(f"  Battery: Unable to read")
            
            # Start video stream
            print("  Starting video stream...")
            self._send_command('streamon')
            time.sleep(3)  # Give stream time to initialize
            
            # Connect to video stream using OpenCV
            video_url = 'udp://0.0.0.0:11111'
            print(f"  Connecting to video at {video_url}...")
            
            self.video_capture = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
            
            if not self.video_capture.isOpened():
                print("⚠ Video stream not ready, will retry during capture")
            else:
                # Set small buffer to reduce latency
                self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print("✓ Video stream connected")
            
            return True
            
        except Exception as e:
            print(f"✗ Drone connection failed: {str(e)}")
            self.is_drone_connected = False
            return False
    
    def _send_command(self, command, timeout=5.0):
        """
        Send command to Tello and wait for response.
        
        Args:
            command (str): Command to send
            timeout (float): Timeout in seconds
            
        Returns:
            str: Response from drone, or None if timeout
        """
        if not self.command_socket:
            return None
        
        try:
            # Send command
            self.command_socket.sendto(
                command.encode('utf-8'),
                ('192.168.10.1', 8889)
            )
            
            # Wait for response
            self.command_socket.settimeout(timeout)
            response, _ = self.command_socket.recvfrom(1024)
            
            return response.decode('utf-8').strip()
            
        except socket.timeout:
            print(f"⚠ Command '{command}' timed out")
            return None
        except Exception as e:
            print(f"⚠ Command '{command}' failed: {e}")
            return None
    
    def _connect_test_video(self):
        """Open test video file for development/testing."""
        try:
            print(f"Opening test video: {config.TEST_VIDEO_PATH}")
            self.video_capture = cv2.VideoCapture(config.TEST_VIDEO_PATH)
            
            if not self.video_capture.isOpened():
                # If test video doesn't exist, create a synthetic feed
                print("⚠ Test video not found, using synthetic feed")
                self.video_capture = None
            else:
                print("✓ Test video loaded")
            
            return True
            
        except Exception as e:
            print(f"✗ Test video failed: {str(e)}")
            return False
    
    def start_stream(self):
        """
        Start continuous frame capture in background thread.
        This keeps the latest frame always available.
        """
        if self.is_streaming:
            print("Stream already running")
            return
        
        self.is_streaming = True
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("✓ Frame capture thread started")
    
    def _capture_loop(self):
        """
        Background thread that continuously captures frames.
        This runs in a loop until stop_stream() is called.
        """
        retry_count = 0
        max_retries = 10
        
        while self.is_streaming:
            try:
                frame = self._get_raw_frame()
                
                if frame is not None:
                    # Update current frame (thread-safe)
                    with self.frame_lock:
                        self.current_frame = frame
                    
                    # Update performance metrics
                    self.frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - self.last_frame_time
                    
                    if elapsed >= 1.0:  # Update FPS every second
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.last_frame_time = current_time
                    
                    retry_count = 0  # Reset retry counter on success
                else:
                    # No frame received
                    retry_count += 1
                    if retry_count >= max_retries and self.use_drone:
                        # Try to reconnect video stream
                        print("⚠ Reconnecting video stream...")
                        self._reconnect_video()
                        retry_count = 0
                
                time.sleep(1.0 / config.DRONE_FPS)  # Control frame rate
                
            except Exception as e:
                print(f"⚠ Frame capture error: {str(e)}")
                time.sleep(0.1)
    
    def _reconnect_video(self):
        """Attempt to reconnect to video stream."""
        try:
            if self.video_capture:
                self.video_capture.release()
            
            time.sleep(1)
            
            if self.use_drone:
                # Reconnect to drone video
                self._send_command('streamoff')
                time.sleep(1)
                self._send_command('streamon')
                time.sleep(2)
                
                self.video_capture = cv2.VideoCapture('udp://0.0.0.0:11111', cv2.CAP_FFMPEG)
                if self.video_capture.isOpened():
                    self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print("✓ Video stream reconnected")
        except:
            pass
    
    def _get_raw_frame(self):
        """
        Get a single frame from drone or test video.
        
        Returns:
            numpy.ndarray: BGR image frame, or None if unavailable
        """
        if self.use_drone and self.video_capture and self.video_capture.isOpened():
            # Get frame from real Tello drone (OpenCV capture)
            ret, frame = self.video_capture.read()
            return frame if ret else None
            
        elif self.video_capture and self.video_capture.isOpened():
            # Get frame from test video
            ret, frame = self.video_capture.read()
            
            if not ret:  # Video ended, loop back to start
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
            
            return frame if ret else None
        
        else:
            # Generate synthetic test frame (checkerboard pattern with random obstacles)
            return self._generate_synthetic_frame()
    
    def _generate_synthetic_frame(self):
        """
        Generate a synthetic overhead view for testing without drone or video.
        Creates a realistic-looking disaster scene with safe and unsafe areas.
        IMPORTANT: Uses fixed seed for consistent obstacles across frames.
        
        Returns:
            numpy.ndarray: Synthetic BGR image
        """
        height, width = config.DRONE_FRAME_HEIGHT, config.DRONE_FRAME_WIDTH
        
        # CRITICAL: Use fixed seed so we get the SAME obstacles every frame!
        # Without this, each frame gets random new obstacles
        np.random.seed(42)
        
        # Start with BRIGHT green safe terrain
        frame = np.ones((height, width, 3), dtype=np.uint8)
        frame[:, :] = (100, 220, 100)  # Bright green = obviously safe
        
        # Calculate positions
        start_x = int(width * 0.10)
        start_y = int(height * 0.10)
        goal_x = int(width * 0.90)
        goal_y = int(height * 0.90)
        
        clearance = 60
        
        # Fixed obstacle pattern - VERY DARK so they're obvious
        obstacles = [
            # Left side
            (int(width * 0.20), int(height * 0.30), 80, 100),
            (int(width * 0.25), int(height * 0.55), 70, 90),
            (int(width * 0.15), int(height * 0.70), 90, 80),
            
            # Middle (forces path planning)
            (int(width * 0.40), int(height * 0.25), 60, 120),
            (int(width * 0.45), int(height * 0.50), 80, 90),
            (int(width * 0.50), int(height * 0.70), 70, 100),
            
            # Right side
            (int(width * 0.65), int(height * 0.30), 75, 110),
            (int(width * 0.70), int(height * 0.55), 85, 95),
            (int(width * 0.75), int(height * 0.75), 80, 90),
            
            # Extra obstacles
            (int(width * 0.35), int(height * 0.15), 50, 50),
            (int(width * 0.55), int(height * 0.40), 45, 55),
            (int(width * 0.60), int(height * 0.85), 50, 45),
        ]
        
        # Draw obstacles in VERY DARK gray (almost black)
        for x, y, w, h in obstacles:
            dist_to_start = np.sqrt((x - start_x)**2 + (y - start_y)**2)
            dist_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            
            if dist_to_start > clearance and dist_to_goal > clearance:
                # BLACK obstacles - impossible to miss!
                cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 20), -1)
                # Add a red border so they're even more obvious
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 180), 3)
        
        # Debris circles - also very dark
        debris = [
            (int(width * 0.32), int(height * 0.12), 25),
            (int(width * 0.42), int(height * 0.85), 30),
            (int(width * 0.58), int(height * 0.20), 28),
            (int(width * 0.78), int(height * 0.45), 22),
            (int(width * 0.18), int(height * 0.60), 26),
        ]
        
        for x, y, r in debris:
            dist_to_start = np.sqrt((x - start_x)**2 + (y - start_y)**2)
            dist_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            
            if dist_to_start > 40 and dist_to_goal > 40:
                cv2.circle(frame, (x, y), r, (25, 25, 25), -1)
                cv2.circle(frame, (x, y), r, (0, 0, 180), 2)
        
        # VERY bright green for start and goal
        cv2.circle(frame, (start_x, start_y), 50, (50, 255, 50), -1)
        cv2.circle(frame, (goal_x, goal_y), 50, (50, 255, 50), -1)
        
        # Add clear labels
        cv2.putText(frame, "START", (start_x - 35, start_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "GOAL", (goal_x - 30, goal_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def get_frame(self):
        """
        Get the most recent frame from the drone.
        This is the main method used by the navigation system.
        
        Returns:
            numpy.ndarray: Latest BGR image frame, or None if not available
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()  # Return a copy for thread safety
            else:
                return None
    
    def get_fps(self):
        """
        Get current frames per second.
        
        Returns:
            float: Current FPS
        """
        return self.fps
    
    def takeoff(self):
        """Command drone to take off (only for real drone)."""
        if self.use_drone and self.is_drone_connected:
            print("Taking off...")
            response = self._send_command('takeoff', timeout=20.0)
            if response == 'ok':
                time.sleep(3)
                print("✓ Drone airborne")
            else:
                print("⚠ Takeoff command may have failed")
    
    def land(self):
        """Command drone to land (only for real drone)."""
        if self.use_drone and self.is_drone_connected:
            print("Landing...")
            response = self._send_command('land', timeout=20.0)
            if response == 'ok':
                print("✓ Drone landed")
            else:
                print("⚠ Land command may have failed")
    
    def emergency_stop(self):
        """Emergency stop - immediately stops all motors."""
        if self.use_drone and self.is_drone_connected:
            print("⚠ EMERGENCY STOP!")
            self._send_command('emergency')
    
    def stop_stream(self):
        """Stop the frame capture thread and cleanup."""
        print("Stopping video stream...")
        self.is_streaming = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.use_drone and self.is_drone_connected:
            try:
                self._send_command('streamoff')
            except:
                pass
        
        if self.video_capture:
            self.video_capture.release()
        
        print("✓ Stream stopped")
    
    def disconnect(self):
        """Disconnect from drone and cleanup all resources."""
        self.stop_stream()
        
        if self.use_drone and self.is_drone_connected:
            try:
                if self.command_socket:
                    self.command_socket.close()
                print("✓ Drone disconnected")
            except:
                pass
        
        self.command_socket = None
        self.video_capture = None
        self.is_drone_connected = False

# Example usage and testing
if __name__ == "__main__":
    print("Testing Drone Input Module (Windows Compatible)")
    print("=" * 50)
    
    # Suppress OpenCV/FFmpeg warnings for cleaner output
    import os
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    
    # Create drone input (will use test mode if drone not available)
    drone_input = DroneInput(use_drone=config.DRONE_ENABLED)
    
    # Connect
    if drone_input.connect():
        # Start streaming
        drone_input.start_stream()
        
        # Capture and display frames for 10 seconds
        print("Capturing frames for 10 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 10:
            frame = drone_input.get_frame()
            
            if frame is not None:
                # Display frame
                cv2.putText(frame, f"FPS: {drone_input.get_fps():.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Drone Feed", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)
        
        cv2.destroyAllWindows()
        
        # Cleanup
        drone_input.disconnect()
        print("Test completed successfully!")
    
    else:
        print("Failed to connect to drone")