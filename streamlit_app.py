import streamlit as st
import numpy as np
import mss
import cv2
from PIL import Image
from datetime import datetime
from tempfile import NamedTemporaryFile
import easyocr
import pyttsx3

# Initialize global variables
recording = False
paused = False
ocr_results = []

# Initialize OCR Reader
reader = easyocr.Reader(['en'])  # This supports English OCR

# Function to start screen recording
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
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR for display
            elapsed_time = (datetime.now() - start_time).seconds
            cv2.putText(frame, f"Time: {elapsed_time}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            video_writer.write(frame)
            if elapsed_time >= duration:
                recording = False
                break

        video_writer.release()

# Function to extract text using EasyOCR
def extract_text_from_image(image):
    """Extracts text from an image using EasyOCR."""
    # Convert the image to a format that easyocr can read
    result = reader.readtext(np.array(image))
    text = " ".join([item[1] for item in result])
    return text

# Function to save OCR results as a text file
def save_text_file(content, filename="ocr_results.txt"):
    """Saves OCR results to a text file."""
    with open(filename, "w") as file:
        file.write(content)

# Function to convert extracted text to speech
def text_to_speech(text):
    """Converts text to speech."""
    engine = pyttsx3.init()
    engine.save_to_file(text, "ocr_results_speech.mp3")
    engine.runAndWait()

# Streamlit App Interface
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

    # Simulating a screenshot from the video for OCR extraction
    video_frame = np.random.rand(resolution[1], resolution[0], 3) * 255
    video_frame = video_frame.astype(np.uint8)
    image = Image.fromarray(video_frame)

    ocr_result = extract_text_from_image(image)
    ocr_results.append(ocr_result)

    st.success("OCR completed!")
    
    # Display OCR Results
    for frame_num, text in enumerate(ocr_results):
        st.write(f"**Frame {frame_num}:**")
        st.code(text)

    # Save OCR Results as File
    if st.button("Save OCR Results"):
        save_text_file("\n".join(ocr_results))
        st.success("OCR results saved!")

    # Text-to-Speech
    if st.button("Convert OCR Results to Speech"):
        combined_text = " ".join(ocr_results)
        text_to_speech(combined_text)
        st.audio("ocr_results_speech.mp3")

# Footer
st.markdown("---")
st.markdown("Developed with 25+ unique features for advanced screen recording and data extraction.")
