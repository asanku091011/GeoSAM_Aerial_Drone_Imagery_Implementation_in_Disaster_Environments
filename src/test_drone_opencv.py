"""
Tello camera using OpenCV instead of PyAV
Works better on Windows
"""
import cv2
import socket
import threading
import time

class TelloOpenCV:
    """Simple Tello interface using OpenCV for video"""
    
    def __init__(self):
        self.command_socket = None
        self.video_socket = None
        self.running = False
        self.frame = None
        
    def connect(self):
        """Connect to Tello"""
        print("Connecting to Tello...")
        
        # Command socket
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_socket.bind(('', 9000))
        
        # Send command mode
        self.send_command('command')
        time.sleep(0.5)
        
        print("✓ Connected!")
        
    def send_command(self, command):
        """Send command to Tello"""
        self.command_socket.sendto(
            command.encode('utf-8'),
            ('192.168.10.1', 8889)
        )
        
    def get_battery(self):
        """Get battery level"""
        self.send_command('battery?')
        try:
            self.command_socket.settimeout(5)
            response, _ = self.command_socket.recvfrom(1024)
            return int(response.decode('utf-8'))
        except:
            return 0
    
    def streamon(self):
        """Start video stream using OpenCV"""
        print("Starting video stream...")
        
        # Tell Tello to start streaming
        self.send_command('streamon')
        time.sleep(2)
        
        # Connect to video stream with OpenCV
        # Tello streams to udp://0.0.0.0:11111
        video_url = 'udp://0.0.0.0:11111'
        
        print(f"Connecting to {video_url}...")
        self.cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            print("✗ Failed to open video stream")
            return False
        
        # Set buffer size
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("✓ Video stream connected!")
        return True
    
    def get_frame(self):
        """Get current frame"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def streamoff(self):
        """Stop video stream"""
        if self.cap:
            self.cap.release()
        self.send_command('streamoff')
    
    def end(self):
        """Disconnect"""
        if self.command_socket:
            self.command_socket.close()

# Main test
if __name__ == "__main__":
    print("="*70)
    print("Tello Camera Test (OpenCV Backend)")
    print("="*70)
    
    drone = TelloOpenCV()
    
    try:
        # Connect
        drone.connect()
        
        # Get battery
        battery = drone.get_battery()
        print(f"Battery: {battery}%")
        
        # Start video
        if drone.streamon():
            print("\nWaiting for first frame (this may take 10-20 seconds)...")
            print("Press 'Q' to quit\n")
            
            got_frame = False
            frame_count = 0
            
            while True:
                frame = drone.get_frame()
                
                if frame is not None:
                    if not got_frame:
                        print("✓ First frame received!")
                        got_frame = True
                    
                    frame_count += 1
                    
                    # Add overlay
                    cv2.putText(frame, f"Battery: {battery}%", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Tello Camera", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if not got_frame:
                        print(".", end='', flush=True)
                    time.sleep(0.1)
        
        # Cleanup
        print("\nClosing...")
        cv2.destroyAllWindows()
        drone.streamoff()
        drone.end()
        print("✓ Done!")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        cv2.destroyAllWindows()
        drone.streamoff()
        drone.end()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()