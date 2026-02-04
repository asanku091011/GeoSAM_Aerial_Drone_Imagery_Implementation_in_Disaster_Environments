from djitellopy import Tello
import cv2
import time
import paramiko
from datetime import datetime

# ===== CONNECT TO TELLO =====
#print("Connecting to Tello...")
#tello = Tello()
#tello.connect()
#print(f"✓ Tello Battery: {tello.get_battery()}%")

#tello.streamon()
#time.sleep(2)
#frame_reader = tello.get_frame_read()

# ===== CONNECT TO RASPBERRY PI =====
print("\nConnecting to Raspberry Pi...")

PI_IP = "10.42.0.1"  # ← Updated to your Pi's actual IP
PI_USERNAME = "pi"
PI_PASSWORD = "raspberry"  # Change if you set a different password

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(PI_IP, username=PI_USERNAME, password=PI_PASSWORD, timeout=10)
    print(f"✓ Connected to Raspberry Pi at {PI_IP}")
except Exception as e:
    print(f"✗ Could not connect to Pi: {e}")
    tello.streamoff()
    tello.end()
    exit()

# ===== MAIN LOOP =====
filename = "tello_current.jpg"

try:
    print("\n=== Starting capture every 5 seconds ===")
    print("Press Ctrl+C to stop\n")
    
    count = 0
    while True:
        count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Capture from Tello
        img = frame_reader.frame
        
        if img is not None:
            # Save image
            cv2.imwrite(filename, img)
            print(f"[{timestamp}] Capture #{count}")
            
            # Get Tello battery
            battery = tello.get_battery()
            print(f"  Tello Battery: {battery}%")
            
            # Control Raspberry Pi
            try:
                # Example: Get Pi's current time
                stdin, stdout, stderr = ssh.exec_command('date')
                pi_time = stdout.read().decode().strip()
                print(f"  Pi time: {pi_time}")
                
                # Example: Get Pi temperature
                stdin, stdout, stderr = ssh.exec_command('vcgencmd measure_temp')
                pi_temp = stdout.read().decode().strip()
                print(f"  Pi temp: {pi_temp}")
                
                # Example: Write to a file on Pi
                log_cmd = f'echo "[{timestamp}] Image #{count} captured" >> /home/pi/tello_log.txt'
                ssh.exec_command(log_cmd)
                
            except Exception as e:
                print(f"  ✗ Pi command error: {e}")
            
            # Optional: Display image
            cv2.imshow("Tello Feed", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
        else:
            print(f"[{timestamp}] No frame from Tello")
        
        print()
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n\nStopping...")

finally:
    print("\nCleaning up...")
    tello.streamoff()
    tello.end()
    ssh.close()
    cv2.destroyAllWindows()
    print("✓ Disconnected from Tello and Pi")