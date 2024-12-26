# Zidio-Project
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from fer import FER
import cv2
from transformers import pipeline
import speech_recognition as sr

# Initialize global objects for processing
sentiment_pipeline = pipeline("sentiment-analysis")
detector = FER(mtcnn=True)

# Directory setup for historical tracking
DATA_DIR = "employee_data"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Real-Time Emotion Detection

def detect_emotion_text(text):
    sentiment = sentiment_pipeline(text)
    return sentiment[0]['label']

def detect_emotion_facial(image_path):
    image = cv2.imread(image_path)
    emotions = detector.detect_emotions(image)
    if emotions:
        return max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
    return "No face detected"

def detect_emotion_speech(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return detect_emotion_text(text)
    except Exception as e:
        return f"Speech recognition error: {e}"

# 2. Task Recommendation

def recommend_tasks(emotion):
    task_map = {
        "POSITIVE": ["Collaborative projects", "Creative brainstorming"],
        "NEGATIVE": ["Relaxed deadlines", "Stress management activities"],
        "NEUTRAL": ["Routine tasks", "Skill development"],
        "happy": ["Team meetings", "Leadership roles"],
        "angry": ["Solo tasks", "Counseling sessions"],
        "sad": ["Low-pressure tasks", "Motivational sessions"],
        "fear": ["Supportive tasks", "Training programs"],
    }
    return task_map.get(emotion.lower(), ["Standard tasks"])

# 3. Historical Mood Tracking

def log_employee_mood(employee_id, emotion):
    file_path = os.path.join(DATA_DIR, f"{employee_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = []

    data.append({"timestamp": datetime.now().isoformat(), "emotion": emotion})

    with open(file_path, "w") as file:
        json.dump(data, file)

# 4. Stress Management Alerts

def stress_alert(emotion):
    negative_emotions = ["angry", "sad", "fear", "NEGATIVE"]
    if emotion.lower() in negative_emotions:
        return "Alert HR: Employee might be stressed or burnt out."
    return "No immediate action required."

# 5. Team Mood Analysis

def analyze_team_mood():
    team_mood = []
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".json"):
            with open(os.path.join(DATA_DIR, file_name), "r") as file:
                data = json.load(file)
                latest_emotion = data[-1]["emotion"] if data else "Unknown"
                team_mood.append(latest_emotion)
    
    mood_counts = pd.Series(team_mood).value_counts().to_dict()
    return mood_counts

# 6. Data Privacy

def anonymize_data():
    anonymized_data = []
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".json"):
            with open(os.path.join(DATA_DIR, file_name), "r") as file:
                data = json.load(file)
                for record in data:
                    anonymized_data.append({"timestamp": record["timestamp"], "emotion": record["emotion"]})
    return anonymized_data

# Example Workflow
if _name_ == "_main_":
    # Simulated input
    employee_id = "E001"
    text_input = "I feel really tired and unmotivated today."
    image_path = "path_to_employee_image.jpg"
    audio_path = "path_to_employee_audio.wav"

    # Detect emotions
    text_emotion = detect_emotion_text(text_input)
    facial_emotion = detect_emotion_facial(image_path)
    speech_emotion = detect_emotion_speech(audio_path)

    # Log emotions
    log_employee_mood(employee_id, text_emotion)

    # Recommendations
    print("Text Emotion:", text_emotion)
    print("Task Recommendation:", recommend_tasks(text_emotion))
    print(stress_alert(text_emotion))

    print("Facial Emotion:", facial_emotion)
    print("Speech Emotion:", speech_emotion)

    # Team Mood Analysis
    print("Team Mood Analysis:", analyze_team_mood())

    # Anonymized Data
    print("Anonymized Data:", anonymize_data())
