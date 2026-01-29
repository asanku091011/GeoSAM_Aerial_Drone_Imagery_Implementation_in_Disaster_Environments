"""
Minimal drone connection test with detailed debugging
"""
import socket
import time

print("="*70)
print("Basic Tello Connection Test")
print("="*70)

# Tello's IP and port
TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
LOCAL_PORT = 9000

print(f"\n[1/4] Creating UDP socket...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', LOCAL_PORT))
    sock.settimeout(5.0)  # 5 second timeout
    print(f"✓ Socket created and bound to port {LOCAL_PORT}")
except Exception as e:
    print(f"✗ Socket error: {e}")
    exit(1)

print(f"\n[2/4] Sending 'command' to drone at {TELLO_IP}:{TELLO_PORT}...")
try:
    message = 'command'
    sock.sendto(message.encode('utf-8'), (TELLO_IP, TELLO_PORT))
    print(f"✓ Sent: '{message}'")
except Exception as e:
    print(f"✗ Send error: {e}")
    sock.close()
    exit(1)

print(f"\n[3/4] Waiting for response (5 second timeout)...")
try:
    response, addr = sock.recvfrom(1024)
    print(f"✓ Received: '{response.decode('utf-8')}' from {addr}")
    print("✓ DRONE IS RESPONDING!")
except socket.timeout:
    print("✗ TIMEOUT - No response from drone")
    print("\nTroubleshooting:")
    print("  1. Are you connected to TELLO-XXXXXX WiFi?")
    print("  2. Is the drone powered on?")
    print("  3. Is the yellow light flashing?")
    print("  4. Try turning drone off and on")
    sock.close()
    exit(1)
except Exception as e:
    print(f"✗ Receive error: {e}")
    sock.close()
    exit(1)

print(f"\n[4/4] Requesting battery level...")
try:
    message = 'battery?'
    sock.sendto(message.encode('utf-8'), (TELLO_IP, TELLO_PORT))
    response, addr = sock.recvfrom(1024)
    battery = response.decode('utf-8')
    print(f"✓ Battery: {battery}%")
except Exception as e:
    print(f"⚠ Battery check failed: {e}")

sock.close()
print("\n" + "="*70)
print("✓ Basic connection test PASSED!")
print("The drone is reachable and responding.")
print("="*70)