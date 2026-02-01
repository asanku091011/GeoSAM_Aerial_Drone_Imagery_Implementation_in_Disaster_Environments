"""
Simple Robot Command Sender (SCP Method - Windows Compatible)
Sends robot commands to Raspberry Pi one at a time using SCP
No additional libraries required!
"""

import time
import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Configuration - CHANGE THESE TO MATCH YOUR SETUP
PI_HOSTNAME = "192.168.1.10"
PI_USERNAME = "asanku"
PI_COMMANDS_DIR = "/home/asanku/Documents/RSEF/movements"
COMMANDS_FILE = "robot_path.txt"  # In outputs folder
DELAY_SECONDS = 5  # Wait 5 seconds between each command

def send_commands_gradually():
    """
    Send commands to Raspberry Pi one at a time.
    Each command is sent with a delay, so the robot can execute them progressively.
    """
    # Find the commands file
    commands_path = os.path.join("outputs", COMMANDS_FILE)
    if not os.path.exists(commands_path):
        commands_path = COMMANDS_FILE
        if not os.path.exists(commands_path):
            print(f"✗ Commands file not found: {COMMANDS_FILE}")
            print("  Make sure you're in the src folder or outputs folder")
            return False
    
    # Read all commands
    print(f"Reading commands from: {commands_path}")
    with open(commands_path, 'r') as f:
        commands = [line.strip() for line in f if line.strip()]
    
    if not commands:
        print("✗ No commands found in file!")
        return False
    
    print(f"\n📡 SENDING {len(commands)} COMMANDS TO RASPBERRY PI")
    print(f"   Target: {PI_USERNAME}@{PI_HOSTNAME}")
    print(f"   Delay: {DELAY_SECONDS} seconds between commands")
    print("=" * 70)
    
    # Show commands
    print("\nCommands to send:")
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
    print()
    
    # Confirm
#    response = input("Proceed? (y/n): ").strip().lower()
#    if response != 'y':
#        print("Cancelled")
#        return False
    
    # Create temp directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temp directory: {temp_dir}")
    
    try:
        # Send each command incrementally
        for i, cmd in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] Sending: {cmd}")
            
            # Create file with all commands up to this point
            temp_file = os.path.join(temp_dir, f"robot_path_{i}.txt")
            with open(temp_file, 'w') as f:
                for c in commands[:i]:
                    f.write(c + '\n')
            
            print(f"            📄 Created temp file with {i} command(s)")
            
            # Build SCP command
            remote_path = f"{PI_USERNAME}@{PI_HOSTNAME}:{PI_COMMANDS_DIR}/{COMMANDS_FILE}"
            scp_cmd = ["scp", temp_file, remote_path]
            
            print(f"            🔄 Uploading to Pi...")
            
            # Execute SCP
            try:
                result = subprocess.run(
                    scp_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print(f"            ✓ Upload successful!")
                else:
                    print(f"            ✗ SCP failed!")
                    print(f"            Error: {result.stderr}")
                    return False
                
            except subprocess.TimeoutExpired:
                print(f"            ✗ SCP timeout (took >30 seconds)")
                return False
            except FileNotFoundError:
                print(f"            ✗ SCP command not found!")
                print(f"            Make sure you have SCP installed (comes with Git for Windows)")
                return False
            
            # Delete temp file
            os.remove(temp_file)
            
            # Wait before next command (except after last one)
            if i < len(commands):
                print(f"            ⏳ Waiting {DELAY_SECONDS} seconds before next command...")
                
                # Show countdown
                for remaining in range(DELAY_SECONDS, 0, -1):
                    print(f"               {remaining}...", end='\r')
                    time.sleep(1)
                print("               " + " " * 20)  # Clear countdown line
        
        print("\n" + "=" * 70)
        print(f"✓ SUCCESS! All {len(commands)} commands sent to Raspberry Pi")
        print(f"  Remote file: {PI_COMMANDS_DIR}/{COMMANDS_FILE}")
        print("=" * 70)
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temp directory
        try:
            os.rmdir(temp_dir)
        except:
            pass

def send_all_at_once():
    """
    Send entire file at once (original behavior).
    """
    commands_path = os.path.join("outputs", COMMANDS_FILE)
    if not os.path.exists(commands_path):
        commands_path = COMMANDS_FILE
        if not os.path.exists(commands_path):
            print(f"✗ Commands file not found: {COMMANDS_FILE}")
            return False
    
    remote_path = f"{PI_USERNAME}@{PI_HOSTNAME}:{PI_COMMANDS_DIR}/{COMMANDS_FILE}"
    
    print(f"\n📡 Sending entire file to Raspberry Pi")
    print(f"   From: {commands_path}")
    print(f"   To:   {remote_path}")
    
    try:
        result = subprocess.run(
            ["scp", commands_path, remote_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✓ File sent successfully!")
            return True
        else:
            print(f"✗ SCP failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def main():
    """Main function."""
    print("=" * 70)
    print("🤖 ROBOT COMMAND SENDER")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Pi Address: {PI_USERNAME}@{PI_HOSTNAME}")
    print(f"  Remote Dir: {PI_COMMANDS_DIR}")
    print(f"  Delay: {DELAY_SECONDS} seconds between commands")
    
    # Check if commands file exists
    commands_path = os.path.join("outputs", COMMANDS_FILE)
    if not os.path.exists(commands_path):
        commands_path = COMMANDS_FILE
    
    if not os.path.exists(commands_path):
        print(f"\n✗ Commands file not found: {COMMANDS_FILE}")
        print("  Please run main.py first to generate robot commands")
        return 1
    
    # Show current commands
    with open(commands_path, 'r') as f:
        commands = [line.strip() for line in f if line.strip()]
    
    print(f"\nCurrent commands ({len(commands)} total):")
    print("-" * 70)
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
    print("-" * 70)
    
    # Menu
    print("\nOptions:")
    print("  1. Send commands ONE AT A TIME with delay (recommended)")
    print("  2. Send entire file at once (fast)")
    print("  3. Cancel")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == '1':
        success = send_commands_gradually()
        return 0 if success else 1
    
    elif choice == '2':
        success = send_all_at_once()
        return 0 if success else 1
    
    else:
        print("\nCancelled")
        return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
