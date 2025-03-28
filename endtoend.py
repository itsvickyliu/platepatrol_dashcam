import gpsd
import cv2
import threading
import time
import os
from picamera2 import Picamera2
from libcamera import Transform
from ultralytics import YOLO
from paddleocr import PaddleOCR
import notecard
from periphery import I2C
import base64
import json
from INA219 import INA219
import logging

def run_inference(infer_frame, inference_dir, det, ocr, frame_count):
    packet = gpsd.get_current()
    start_time = time.time()
    start = start_time
    results = det(infer_frame)
    latency = time.time() - start
    print(f"Detection latency: {latency:.3f} seconds")
    for i, result in enumerate(results[0].boxes):
        print(f"{len(results[0].boxes)} license plates detected")
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        # Crop the detected region (license plate)
        cropped_img = infer_frame[y1:y2, x1:x2]

        # Run PaddleOCR on the cropped image
        start = time.time()
        ocr_result = ocr.ocr(cropped_img, cls=False, det=False)
        latency = time.time() - start
        print(f"OCR latency: {latency:.3f} seconds")
        if ocr_result:
            recognized_text = ''.join([line[0][0] for line in ocr_result])
            # Ask server if this license plate is a hit
            req = {"req": "web.get"}
            req["route"] = "GetDetection"
            req["name"] = f'/{recognized_text}'
            req["content"] = "plain/text"
            start = time.time()
            rsp = card.Transaction(req)
            latency = time.time() - start
            print(req)
            print(f"Network handshake latency: {latency:.3f} seconds")

            if rsp["result"] == 200:
                # Decode plaintext response to JSON
                encoded_payload = rsp["payload"]
                decoded_payload = base64.b64decode(encoded_payload).decode('utf-8')
                json_payload = json.loads(decoded_payload)

                # Transfer target license plate image to server with rate limiting
                # if json_payload.get("match") and rsp["body"].get("X-From-Cache") is None:
                    # start = time.time()
                cropped_filename = os.path.join(inference_dir, f"inf_{frame_count}_{i}.jpg")
                print(f"inf_{frame_count}_{i}.jpg: {recognized_text}, lon: {packet.lon}, lat: {packet.lat}, timestamp: {start_time}")
                cv2.imwrite(cropped_filename, cropped_img)
                    # latency = time.time() - start
                    # print(f"Image transfer latency: {latency:.3f} seconds")
                    # latency = time.time() - start_time
                    # print(f"Critical path latency: {latency:.3f} seconds")

def main_loop(inference_dir, raw_dir, model, ocr):
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
    
    # Create output directories if they don't exist
    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Generate a unique timestamp for each recording
    timestamp = time.time()
        
    # Create new subdirectories for raw and inference frames based on timestamp
    timestamp_inference_dir = os.path.join(inference_dir, f"{timestamp}")
    timestamp_raw_dir = os.path.join(raw_dir, f"{timestamp}")
    os.makedirs(timestamp_inference_dir, exist_ok=True)
    os.makedirs(timestamp_raw_dir, exist_ok=True)
    
    # Setup for raw video recording with MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_duration = 60  # seconds per clip
    start_time = time.time()
    clip_index = 0

    # Capture an initial frame to determine dimensions
    frame = picam2.capture_array()
    frame_count = 0
    width = 640
    height = 480
    fps = 12  # Initial FPS assumption
    raw_filename = os.path.join(timestamp_raw_dir, f"raw_clip_{clip_index}.mp4")
    out = cv2.VideoWriter(raw_filename, fourcc, fps, (width, height))
    
    inference_thread = None

    discharge_counter = 0

    while True:
        bus_voltage = ina219.getBusVoltage_V()             # voltage on V- (load side)
        shunt_voltage = ina219.getShuntVoltage_mV() / 1000 # voltage between V+ and V- across the shunt
        current = ina219.getCurrent_mA() / 1000                 # current in mA
        power = ina219.getPower_W()                        # power in W
        p = (bus_voltage - 6)/2.4*100
        if(p > 100):p = 100
        if(p < 0):p = 0
        if current < -0.5:
            status = "Discharging"
            discharge_counter += 1
            print("Load Voltage:  {:6.3f} V".format(bus_voltage))
            print("Current:       {:9.6f} A".format(current))
            print("Power:         {:6.3f} W".format(power))
            print("Percent:       {:3.1f}%".format(p))
            print("")
        else:
            discharge_counter = 0

        # Capture current frame
        frame = picam2.capture_array()
        frame_count += 1
        
        # Write the resized frame to the current MP4 file
        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)

        # Calculate and display actual FPS every second
        if frame_count % fps == 0:
            elapsed_time = time.time() - start_time
            actual_fps = frame_count / elapsed_time
            print(f"Elapsed Time: {elapsed_time:.2f}s, Frames Captured: {frame_count}, Actual FPS: {actual_fps:.2f}")
        
        # Check if it's time to start a new clip
        if time.time() - start_time >= clip_duration:
            out.release()
            clip_index += 1
            start_time = time.time()
            frame_count = 0  # Reset frame count for the new clip
            # Recalculate FPS based on previous clip's performance
            fps = round(actual_fps) if 'actual_fps' in locals() else fps
            raw_filename = os.path.join(timestamp_raw_dir, f"raw_clip_{clip_index}.mp4")
            out = cv2.VideoWriter(raw_filename, fourcc, fps, (width, height))
        
        # Only start a new inference if the previous one has finished
        if inference_thread is None or not inference_thread.is_alive():
            # Pass a copy of the frame for inference to avoid interference with recording
            inference_thread = threading.Thread(
                target=run_inference,
                args=(frame.copy(), timestamp_inference_dir, model, ocr, frame_count)
            )
            inference_thread.start()
        
        if cv2.waitKey(1) & 0xFF == ord('q') or discharge_counter >= 5:
            break

    # Release resources on exit
    out.release()
    picam2.stop()
    cv2.destroyAllWindows()

# YOLO setup
det_model = YOLO("detection_ncnn_model")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# OCR setup
ocr_model = PaddleOCR(
    det_model_dir=None,
    cls_model_dir=None,
    rec_model_dir='ocr',
    rec_algorithm='SVTR_LCNet',
    rec_char_dict_path='/home/platepatrol/Desktop/PlatePatrol/ocr/character_dict.txt',
    use_angle_cls=True,
    lang='en'
)

# Cellular setup
productUID = "edu.cmu.andrew.ziyaoz:platepatrol"
port = I2C("/dev/i2c-1")

card = notecard.OpenI2C(port, 0, 0, debug=True)
print("Connected to Notecard...")

req = {"req": "hub.status"}
while True:
    rsp = card.Transaction(req)
    if rsp.get("connected"):
        break
    time.sleep(1)

req = {"req": "hub.set"}
req["product"] = productUID
req["mode"] = "continuous"
card.Transaction(req)

# GPS setup
gpsd.connect()
while True:
    packet = gpsd.get_current()
    if packet.mode >= 2:  # Check for 2D or 3D fix
        print("GPS connected")
        break
    else:
        time.sleep(1)

# UPS setup
ina219 = INA219(addr=0x42)

main_loop('/home/platepatrol/Desktop/inference_frames', '/home/platepatrol/Desktop/raw_footage', det_model, ocr_model)
