import gpsd
import cv2
import threading
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR
import notecard
from periphery import I2C
from INA219 import INA219
import RPi.GPIO as GPIO
import logging
from datetime import datetime

def run_inference(infer_frame, det, ocr):
    lat = 40.443428
    lon = -79.939505
    get_gps_data_with_timeout(0.005, 1)
    if packet:
        lat = packet.lat
        lon = packet.lon
        
    gps = f"{lat},{lon}"

    start_time = time.time()
    results = det(infer_frame)
    latency = time.time() - start_time

    print(f"Detection latency: {latency:.3f} seconds")
    print(f"{len(results[0].boxes)} license plates detected")

    for i, result in enumerate(results[0].boxes):
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
                color = (0, 255, 0) if recognized_text in correct_plates else (0, 0, 255)

                # Draw bounding box and text
                cv2.rectangle(infer_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(infer_frame, recognized_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

     # Prepare the text to be displayed
    readable_timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    display_text = f"GPS: {gps} | Timestamp: {readable_timestamp} | Latency: {time.time() - start_time:.3f} seconds"

    # Overlay the text on the frame
    cv2.putText(infer_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(infer_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)   
    cv2.imshow("Inference", infer_frame)

def main_loop(model, ocr, video_path):
    global led_state
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        timestamp = time.time()

        discharge_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video from the beginning
                continue

            bus_voltage = ina219.getBusVoltage_V()
            current = ina219.getCurrent_mA() / 1000
            power = ina219.getPower_W()
            p = (bus_voltage - 6) / 2.4 * 100
            p = max(0, min(100, p))
            
            if current < -0.5:
                status = "Discharging"
                discharge_counter += 1
                print(f"Load Voltage: {bus_voltage:.3f} V, Current: {current:.6f} A, Power: {power:.3f} W, Percent: {p:.1f}%, Time: {time.time()}")

            else:
                discharge_counter = 0
            
            if GPIO.input(OPTIN_PIN):
                led_state = "inferencing"
                run_inference(frame, model, ocr)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or discharge_counter >= 5:
                break

    except Exception as e:
        print(f"Error encountered: {e}")
    
    finally:
        print("Exiting... Turning off LED and cleaning up resources.")
        led_state = "done"
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
GPIO.setup(OPTIN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# YOLO setup
det_model = YOLO("detection_ncnn_model")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# OCR setup
ocr_model = PaddleOCR(
    det_model_dir=None,
    cls_model_dir=None,
    rec_model_dir='ocr_model',
    rec_algorithm='SVTR_LCNet',
    rec_char_dict_path='/home/platepatrol/Desktop/PlatePatrol/ocr_model/character_dict.txt',
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
    global packet
    for i in range(max_attempts):
        gps_thread = threading.Thread(target=get_gps_data, daemon=True)
        gps_thread.start()
    
        gps_thread.join(timeout_seconds)

        if gps_thread.is_alive():
            print(f"Attempt {i}: GPS data request timed out.")
            packet = None
            gps_thread.join()
        elif packet and packet.mode >= 2:
            print("GPS connected")
            break
        else:
            print(f"Attempt {i}: No GPS fix, retrying...")
            packet = None

# GPS setup
gpsd.connect()
packet = None
get_gps_data_with_timeout(1, 3)

# UPS setup
ina219 = INA219(addr=0x42)

correct_plates = [
    "8CZ9535", "SC80343", "ZWY2410", "LDL9870", "MLE1167", "GRT2919",
    "47122", "MT47122", "1BADAXE", "LYZ3974", "HZJ5075", "MPL5975",
    "HRV3624", "HKV3624", "ZMA1423", "MHA2335", "LJN9478", "LVR1387",
    "LWC6315", "VULPINE", "DE40814", "48444", "MT48444", "51437",
    "MT51437", "MBF8905", "LSS9419", "KOR3240", "MSP1682", "HYT8681",
    "LBC1224", "KDY1547", "ZXD6096", "MPM7112", "72CL", "BR72CL",
    "LNT1141", "FFR8167", "MPY5179", "MSS4948", "LVN3425", "LYD2221",
    "LKS4291", "LMD2231", "DGU748", "KBS3573", "LZA7658", "4FD8385"
]

main_loop(det_model, ocr_model, "/home/platepatrol/Desktop/raw_highres_footage/1744306803.0886111/raw_clip_3.mp4")