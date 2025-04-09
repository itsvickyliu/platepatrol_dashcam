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
import RPi.GPIO as GPIO
import logging

def run_inference(infer_frame, inference_dir, det, ocr, frame_count):
    start_time = time.time()

    lat = 0
    lon = 0
    get_gps_data_with_timeout(0.005, 1)
    if packet:
        lat = packet.lat
        lon = packet.lon
        
    
    gps = f"{lat},{lon}"

    start = time.time()
    results = det(infer_frame)
    latency = time.time() - start

    print(f"Detection latency: {latency:.3f} seconds")
    print(f"{len(results[0].boxes)} license plates detected")

    for i, result in enumerate(results[0].boxes):
        if result.conf < 0.5:
            continue
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        # Crop the detected region (license plate)
        cropped_img = infer_frame[y1:y2, x1:x2]

        # Skip OCR if height >= width
        if cropped_img.shape[0] >= cropped_img.shape[1]:
            continue

        # Run PaddleOCR on the cropped image
        start = time.time()
        ocr_result = ocr.ocr(cropped_img, cls=False, det=False)
        latency = time.time() - start
        print(f"OCR latency: {latency:.3f} seconds")
        if ocr_result:
            recognized_text = ''.join([line[0][0] for line in ocr_result])

            if len(recognized_text) >= 2 and len(recognized_text) <= 8:
                # Ask server if this license plate is a hit
                req = {"req": "web.get"}
                req["route"] = "GetDetection"
                req["name"] = f'/{recognized_text}'
                req["content"] = "plain/text"
                start = time.time()
                rsp = card.Transaction(req)
                latency = time.time() - start
                print(f"Network handshake latency: {latency:.3f} seconds")

                if rsp["result"] == 200:
                    # Decode plaintext response to JSON
                    encoded_payload = rsp["payload"]
                    decoded_payload = base64.b64decode(encoded_payload).decode('utf-8')
                    json_payload = json.loads(decoded_payload)

                    # For debugging purposes, store all inferenced images
                    cropped_filename = os.path.join(inference_dir, f"inf_{frame_count}_{i}_{recognized_text}.jpg")
                    print(f"inf_{frame_count}_{i}.jpg: {recognized_text}, lat: {packet.lat}, lon: {packet.lon}, timestamp: {start_time}")
                    cv2.imwrite(cropped_filename, cropped_img)

                    # Transfer target license plate image to server with rate limiting
                    if json_payload.get("match") and rsp["body"].get("X-From-Cache") is None:
                        image_id = json_payload.get("image_id")
                        print(f"id: {image_id}")
                        success, encoded_image = cv2.imencode(".jpg", cropped_img)
                        if success:
                            chunk_size = 8 * 1024

                            image_buffer = encoded_image.tobytes()

                            chunks = [image_buffer[i:i + chunk_size] for i in range(0, len(image_buffer), chunk_size)]
                            total_chunks = len(chunks)

                            start = time.time()
                            for i, chunk in enumerate(chunks):
                                encoded_chunk = base64.b64encode(chunk).decode('utf-8')
                                req = {"req": "web.post"}
                                req["route"] = "PostChunk"
                                if i == 0:
                                    req["body"] = {"image_id": image_id, "chunk_id": i, "total_chunks": total_chunks, "data": encoded_chunk, "gps_location": gps, "timestamp": start_time}
                                else:
                                    req["body"] = {"image_id": image_id, "chunk_id": i, "total_chunks": total_chunks, "data": encoded_chunk}
                                req["content"] = "plain/text"
                                rsp = card.Transaction(req)
                            latency = time.time() - start
                            print(f"Image transfer latency: {latency:.3f} seconds")
                            latency = time.time() - start_time
                            print(f"Critical path latency: {latency:.3f} seconds")

def main_loop(inference_dir, raw_dir, model, ocr):
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
        
        os.makedirs(inference_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)

        timestamp = time.time()
        timestamp_inference_dir = os.path.join(inference_dir, f"{timestamp}")
        timestamp_raw_dir = os.path.join(raw_dir, f"{timestamp}")
        os.makedirs(timestamp_inference_dir, exist_ok=True)
        os.makedirs(timestamp_raw_dir, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        clip_duration = 60  
        start_time = time.time()
        clip_index = 0

        frame = picam2.capture_array()
        frame_count = 0
        width, height = 640, 480
        fps = 12  
        raw_filename = os.path.join(timestamp_raw_dir, f"raw_clip_{clip_index}.mp4")
        out = cv2.VideoWriter(raw_filename, fourcc, fps, (width, height))
        
        inference_thread = None
        discharge_counter = 0

        while True:
            if inference_thread is None or not inference_thread.is_alive():
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
            
            if inference_thread is None or not inference_thread.is_alive() and GPIO.input(OPTIN_PIN):
                led_state = "inferencing"
                inference_thread = threading.Thread(
                    target=run_inference,
                    args=(frame.copy(), timestamp_inference_dir, model, ocr, frame_count)
                )
                inference_thread.start()
            
            if cv2.waitKey(1) & 0xFF == ord('q') or discharge_counter >= 5:
                break

    except Exception as e:
        print(f"Error encountered: {e}")
    
    finally:
        if inference_thread:
            inference_thread.join()
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
        elif led_state == "inferencing":
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.1)
        elif led_state == "done":
            GPIO.output(LED_PIN, GPIO.LOW)
            GPIO.cleanup()

# Start LED control in a separate thread
led_thread = threading.Thread(target=led_control, daemon=True)
led_thread.start()

# Opt-in switch setup
OPTIN_PIN = 17    # GPIO number

GPIO.setmode(GPIO.BCM)
GPIO.setup(OPTIN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

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

def get_gps_data():
    global packet
    try:
        packet = gpsd.get_current()
    except Exception as e:
        print(f"Error getting GPS data: {e}")
        packet = None

def get_gps_data_with_timeout(timeout_seconds, max_attempts):
    for i in range(max_attempts):
        gps_thread = threading.Thread(target=get_gps_data, daemon=True)
        gps_thread.start()
    
        gps_thread.join(timeout_seconds)

        if gps_thread.is_alive():
            print(f"Attempt {i}: GPS data request timed out.")
            gps_thread.join()
        elif packet and packet.mode >= 2:
            print("GPS connected")
            break
        else:
            print(f"Attempt {i}: No GPS fix, retrying...")

# GPS setup
gpsd.connect()
packet = None
get_gps_data_with_timeout(1, 3)

# UPS setup
ina219 = INA219(addr=0x42)

main_loop('/home/platepatrol/Desktop/inference_frames', '/home/platepatrol/Desktop/raw_footage', det_model, ocr_model)