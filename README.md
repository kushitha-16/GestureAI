# 🤖 GestureAI Pro – ML-Based Gesture and Voice Controlled Desktop Assistant

## 🚀 Overview

GestureAI Pro is a real-time Computer Vision and Machine Learning desktop assistant that enables touchless interaction with a computer using hand gestures and voice commands.

The project combines **OpenCV**, **MediaPipe**, **Scikit-learn**, **Speech Recognition**, and **PyAutoGUI** to create an intelligent Human-Computer Interaction (HCI) system capable of controlling desktop applications without using a keyboard or mouse.

A custom Machine Learning model is trained on hand landmark data to recognize gestures and perform desktop automation tasks such as air mouse control, mouse clicking, volume control, application launching, browser navigation, and voice command execution.

---

# ✨ Features

### 🤖 Machine Learning

* Custom gesture dataset collection
* K-Nearest Neighbors (KNN) gesture classifier
* Real-time ML gesture prediction
* Multiple gesture recognition

### 🖐 Computer Vision

* Real-time hand tracking
* 21 hand landmark detection using MediaPipe
* Gesture recognition using ML
* Live webcam processing

### 🖱 Desktop Automation

* Air Mouse
* Pinch Click
* Volume Up
* Volume Down
* Play / Pause Media
* Open VS Code
* Open GitHub
* Open YouTube

### 🎙 Voice Assistant

* Speech Recognition
* Text-to-Speech
* Voice-controlled browser search
* Application launching
* Desktop commands

---

# 🛠 Technology Stack

## Programming Language

* Python

## Machine Learning

* Scikit-learn (KNN)
* Joblib

## Computer Vision

* OpenCV
* MediaPipe

## Speech Processing

* SpeechRecognition
* Pyttsx3

## Desktop Automation

* PyAutoGUI

## Supporting Libraries

* NumPy
* Pandas

---

# 🎯 Supported Gestures

| Gesture          | Action                   |
| ---------------- | ------------------------ |
| ✋ Palm           | Activate Voice Assistant |
| ✌ Victory        | Open VS Code             |
| ☝ One Finger     | Air Mouse                |
| 👌 OK Sign       | Left Click               |
| 👍 Thumbs Up     | Volume Up                |
| 🖐 Three Fingers | Volume Down              |
| 🤘 Rock          | Play / Pause Media       |
| 🤙 Call Me       | Open YouTube             |
| ☮ Peace          | Open GitHub              |
| ✊ Fist           | Exit Application         |

---

# 📂 Project Structure

```text
GestureAI/
│
├── main.py
├── gesture_ai_pro_ml.py
├── collect_data.py
├── train_model.py
├── ml_gesture_ai.py
├── air_mouse.py
├── direction_control.py
├── gesture_data.csv
├── gesture_model.pkl
├── requirements.txt
├── README.md
└── screenshots/
```

---

# ⚙ System Workflow

```
Webcam
   │
   ▼
MediaPipe Hand Detection
   │
   ▼
21 Hand Landmarks
   │
   ▼
Machine Learning Model (KNN)
   │
   ▼
Gesture Prediction
   │
   ▼
Action Mapping
   │
   ▼
Desktop Automation & Voice Assistant
```

---

# 🧠 Machine Learning Pipeline

1. Collect hand landmark data
2. Store landmarks in CSV dataset
3. Train KNN gesture classifier
4. Save trained model
5. Load model for real-time prediction
6. Execute corresponding desktop action

---

# 📸 Current Capabilities

✅ Real-time Hand Tracking

✅ Machine Learning Gesture Recognition

✅ Air Mouse

✅ Pinch Click

✅ Volume Control

✅ Voice Assistant

✅ Browser Automation

✅ VS Code Launcher

✅ Desktop Automation

---

# 💻 Installation

```bash
git clone https://github.com/kushitha-16/GestureAI.git

cd GestureAI

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt

python gesture_ai_pro_ml.py
```

---

# 🎓 Concepts Demonstrated

* Computer Vision
* Machine Learning
* Human Computer Interaction (HCI)
* Gesture Recognition
* Real-Time Prediction
* Speech Recognition
* Desktop Automation
* Voice Processing
* Data Collection
* Model Training

---

# 🔮 Future Enhancements

* Deep Learning Gesture Recognition
* CNN-based Gesture Classification
* Sign Language Recognition
* Multi-Hand Tracking
* AI Screen Assistant
* Face Recognition Login
* Smart Home Automation
* IoT Device Control
* Gesture Customization

---

# 👨‍💻 Author

**Maddirevula Kushitha**

Bachelor of Engineering – Computer Science

Dayananda Sagar Academy of Technology and Management

**GitHub:** https://github.com/kushitha-16

---

