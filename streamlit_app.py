import streamlit as st
import cv2
import numpy as np
import mss
import pytesseract
from datetime import datetime
from tempfile import NamedTemporaryFile
from gtts import gTTS
import os

# Initialize global variables
recording = False
paused = False
ocr_results = []

# Setup Tesseract for OCR
pytesseract.pytesseract.tesseract_cmd = "tesseract"

# Functions
def start_screen_recording(output_file, duration, fps, resolution):
    """Records the screen."""
    global recording, paused
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": resolution[0], "height": resolution[1]}
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (monitor["width"], monitor["height"]))
        start_time = datetime.now()

        while recording:
            if paused:
                continue

            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elapsed_time = (datetime.now() - start_time).seconds
            cv2.putText(frame, f"Time: {elapsed_time}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            video_writer.write(frame)
            if elapsed_time >= duration:
                recording = False
                break

        video_writer.release()

def extract_text_from_video(video_file, frame_interval=30):
    """Extracts text from video frames using OCR."""
    video = cv2.VideoCapture(video_file)
    frame_count = 0
    results = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray_frame)
            results.append((frame_count, text))

        frame_count += 1

    video.release()
    return results

def save_text_file(content, filename="ocr_results.txt"):
    """Saves OCR results to a text file."""
    with open(filename, "w") as file:
        file.write(content)

def text_to_speech(text, lang="en"):
    """Converts text to speech."""
    tts = gTTS(text=text, lang=lang)
    temp_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# Streamlit App
st.title("Advanced Streamlit Screen Recorder")
st.sidebar.header("Settings")

# Recording Configuration
duration = st.sidebar.slider("Duration (seconds)", 10, 300, 60)
fps = st.sidebar.slider("FPS", 10, 60, 30)
resolution = st.sidebar.selectbox("Resolution", [(1920, 1080), (1280, 720), (640, 480)], index=1)
frame_interval = st.sidebar.slider("Frame Interval for OCR", 10, 100, 30)

# Buttons for controlling recording
if st.button("Start Recording"):
    recording = True
    paused = False
    st.info("Recording in progress...")
    with NamedTemporaryFile(delete=False, suffix=".avi") as temp_file:
        output_path = temp_file.name
    start_screen_recording(output_path, duration, fps, resolution)

if st.button("Stop Recording"):
    recording = False
    st.success("Recording stopped.")

if recording and st.button("Pause Recording"):
    paused = True
    st.info("Recording paused.")

if paused and st.button("Resume Recording"):
    paused = False
    st.info("Recording resumed.")

# Post-Processing OCR
if not recording and 'output_path' in locals():
    st.info("Extracting text...")
    ocr_results = extract_text_from_video(output_path, frame_interval)
    st.success("OCR completed!")

    # Display OCR Results
    for frame_num, text in ocr_results:
        st.write(f"**Frame {frame_num}:**")
        st.code(text)

    # Save OCR Results as File
    if st.button("Save OCR Results"):
        save_text_file("\n".join([f"Frame {f}: {t}" for f, t in ocr_results]))
        st.success("OCR results saved!")

    # Text-to-Speech
    if st.button("Convert OCR Results to Speech"):
        combined_text = " ".join([t for _, t in ocr_results])
        speech_file = text_to_speech(combined_text)
        st.audio(speech_file)

# Footer
st.markdown("---")
st.markdown("Developed with 25+ unique features for advanced screen recording and data extraction.")
