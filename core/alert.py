import threading
import time
import subprocess
import os

class AlertManager:
    def __init__(self, sound_path="assets/alert.wav", cooldown=3.0):
        self.sound_path = sound_path
        self.cooldown = cooldown
        self.last_alert_time = 0.0
        self.is_playing = False

    def play_sound_thread(self):
        try:
            if os.path.exists(self.sound_path):
                subprocess.run(["afplay", self.sound_path])
            else:
                print(f"[Warning] Sound file not found: {self.sound_path}")
        except Exception as e:
            print(f"[Error] Failed to play sound: {e}")
        finally:
            self.is_playing = False

    def trigger(self):
        current_time = time.time()
        # Only trigger if cooldown has passed and we aren't already playing
        if current_time - self.last_alert_time > self.cooldown and not self.is_playing:
            self.last_alert_time = current_time
            self.is_playing = True
            thread = threading.Thread(target=self.play_sound_thread, daemon=True)
            thread.start()
