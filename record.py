import cv2
import threading
import time
import os
from picamera2 import Picamera2
from libcamera import Transform
from periphery import I2C
from INA219 import INA219
import RPi.GPIO as GPIO

def main_loop(raw_dir):
    global led_state
    try:
        # Initialize Picam2
        picam2 = Picamera2()
        picam2.video_configuration.main.size = (2304, 1296)
        picam2.video_configuration.main.format = "RGB888"
        picam2.video_configuration.align()
        picam2.video_configuration.controls.FrameDurationLimits = (33333, 33333)
        picam2.video_configuration.controls.AeExposureMode = 1 # ExposureShort
        picam2.video_configuration.transform = Transform(vflip=True, hflip=True)
        
        picam2.configure("video")
        picam2.start()
        
        os.makedirs(raw_dir, exist_ok=True)

        timestamp = time.time()
        timestamp_raw_dir = os.path.join(raw_dir, f"{timestamp}")
        os.makedirs(timestamp_raw_dir, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        clip_duration = 60 * 5 
        start_time = time.time()
        clip_index = 0

        frame = picam2.capture_array()
        frame_count = 0
        width, height = 2304, 1296
        fps = 16  
        raw_filename = os.path.join(timestamp_raw_dir, f"raw_clip_{clip_index}.mp4")
        out = cv2.VideoWriter(raw_filename, fourcc, fps, (width, height))
        
        discharge_counter = 0

        while True:
            led_state = "recording"
            bus_voltage = ina219.getBusVoltage_V()
            current = ina219.getCurrent_mA() / 1000
            power = ina219.getPower_W()
            p = (bus_voltage - 6) / 2.4 * 100
            p = max(0, min(100, p))
            
            if current < -0.5:
                status = "Discharging"
                discharge_counter += 1
                print(f"Load Voltage: {bus_voltage:.3f} V, Current: {current:.6f} A, Power: {power:.3f} W, Percent: {p:.1f}%")

            else:
                discharge_counter = 0

            frame = picam2.capture_array()
            frame_count += 1
            resized_frame = cv2.resize(frame, (width, height))
            out.write(resized_frame)

            if fps > 0 and frame_count % fps == 0:
                elapsed_time = time.time() - start_time
                actual_fps = frame_count / elapsed_time
                print(f"Elapsed Time: {elapsed_time:.2f}s, Frames Captured: {frame_count}, Actual FPS: {actual_fps:.2f}")
            
            if time.time() - start_time >= clip_duration:
                out.release()
                clip_index += 1
                start_time = time.time()
                frame_count = 0  
                fps = round(actual_fps) if 'actual_fps' in locals() else fps
                raw_filename = os.path.join(timestamp_raw_dir, f"raw_clip_{clip_index}.mp4")
                out = cv2.VideoWriter(raw_filename, fourcc, fps, (width, height))
            
            if cv2.waitKey(1) & 0xFF == ord('q') or discharge_counter >= 5:
                break

    except Exception as e:
        print(f"Error encountered: {e}")
    
    finally:
        print("Exiting... Turning off LED and cleaning up resources.")
        led_state = "done"
        out.release()
        picam2.stop()
        cv2.destroyAllWindows()
        # Ensure LED state is updated
        time.sleep(1)

# LED setup
LED_PIN = 23    # GPIO number

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Global flag for LED control state
led_state = "setup"

def led_control():
    while led_state != "done":
        if led_state == "setup":
            GPIO.output(LED_PIN, GPIO.HIGH)  # LED stays ON
        elif led_state == "recording":
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.5)
        elif led_state == "done":
            GPIO.output(LED_PIN, GPIO.LOW)
            GPIO.cleanup()

# Start LED control in a separate thread
led_thread = threading.Thread(target=led_control, daemon=True)
led_thread.start()

# UPS setup
ina219 = INA219(addr=0x42)

main_loop('/home/platepatrol/Desktop/raw_highres_footage')