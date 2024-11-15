import cv2
import numpy as np
import pyautogui
import time
import streamlit as st
from PIL import Image
from datetime import datetime

# Helper class for screen recording
class ScreenRecorder:
    def __init__(self, fps=15, duration=10, resolution=(1920, 1080)):
        self.fps = fps
        self.duration = duration
        self.resolution = resolution
        self.video_writer = None
        self.start_time = None

    def init_recorder(self):
        # Initialize the video writer
        filename = f"screen_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), self.fps, self.resolution)
        self.start_time = time.time()
        return filename

    def record_activity(self, url, duration):
        # Perform screen capture and record the activity
        print(f"Recording started for URL: {url} for {duration} seconds...")
        
        while time.time() - self.start_time < duration:
            # Capture the screen
            screenshot = pyautogui.screenshot(region=(0, 0, *self.resolution))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)
            time.sleep(1 / self.fps)
        
        print(f"Recording finished! Saved as {self.init_recorder()}")
        return self.init_recorder()

    def stop_recorder(self):
        if self.video_writer:
            self.video_writer.release()
            print("Recorder stopped and file saved.")

# Streamlit UI
def main():
    st.title('Screen Recording Plugin')

    # Display instructions
    st.write("""
        This plugin allows you to record your screen and save the video. You can specify the duration,
        frames per second (FPS), and resolution for recording. It uses Python's OpenCV and PyAutoGUI for
        capturing and saving the screen.
    """)

    # Input parameters
    duration = st.number_input("Duration (seconds)", min_value=1, value=10, step=1)
    fps = st.slider("Frames per Second (FPS)", min_value=1, max_value=30, value=15)
    resolution = (1920, 1080)  # You can adjust or add more dynamic resolution options

    # Start button for recording
    if st.button('Start Recording'):
        recorder = ScreenRecorder(fps=fps, duration=duration, resolution=resolution)
        
        # Start recording
        filename = recorder.init_recorder()
        video_file = recorder.record_activity("example.com", duration)
        
        # Stop recording and show output
        recorder.stop_recorder()

        # Show the video file link in Streamlit
        st.success(f"Recording completed! Video saved as {video_file}")
        st.video(video_file)

if __name__ == "__main__":
    main()
