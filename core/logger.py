import cv2
import csv
import os
import datetime

class EventLogger:
    def __init__(self, log_dir="logs", image_dir="captured_images", log_file="events.csv"):
        self.log_dir = log_dir
        self.image_dir = image_dir
        self.log_file = os.path.join(self.log_dir, log_file)
        
        # Ensure directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Event", "Image Path"])

    def log_violation(self, frame, event_msg="Gloves Not Detected"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        image_name = f"violation_{timestamp}.jpg"
        image_path = os.path.join(self.image_dir, image_name)
        
        # Save image
        cv2.imwrite(image_path, frame)
        
        # Write to log
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event_msg, image_path])
            
        print(f"[Log] Violation recorded at {timestamp}")
