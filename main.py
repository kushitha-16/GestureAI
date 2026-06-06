import cv2
import mediapipe as mp
import pyautogui
import os
import time
import speech_recognition as sr
import pyttsx3
import webbrowser
import google.generativeai as genai
from dotenv import load_dotenv

# =========================
# GEMINI SETUP
# =========================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
else:
    model = None

# =========================
# TEXT TO SPEECH
# =========================

engine = pyttsx3.init()

def speak(text):
    print("Assistant:", text)
    engine.say(text[:250])
    engine.runAndWait()

# =========================
# SPEECH RECOGNITION
# =========================

recognizer = sr.Recognizer()

def listen_command():
    with sr.Microphone() as source:
        speak("Listening for your command")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(
                source,
                timeout=6,
                phrase_time_limit=8
            )

            command = recognizer.recognize_google(audio)
            command = command.lower()

            print("You said:", command)
            return command

        except sr.WaitTimeoutError:
            speak("No voice detected")
            return ""

        except sr.UnknownValueError:
            speak("Sorry, I could not understand")
            return ""

        except sr.RequestError:
            speak("Speech service error")
            return ""

# =========================
# GEMINI AI RESPONSE
# =========================

def ask_gemini(command):

    if model is None:
        return "Gemini API key is missing. Please check your .env file."

    prompt = f"""
You are GestureAI, an AI-powered computer assistant.

User command: "{command}"

Rules:
1. If user wants to open a website, return exactly:
OPEN_URL: website_url

2. If user wants to search something, return exactly:
SEARCH: search_query

3. If user asks questions like interview questions, explanations, roadmap, or preparation tips,
give a short useful answer in simple words.

4. Keep answer short and clear.
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Gemini error: {e}"

# =========================
# COMMAND EXECUTION
# =========================

def execute_voice_command(command):

    if command == "":
        return

    # Direct commands first

    if "open chrome" in command:
        speak("Opening Chrome")
        os.system("start chrome")
        return

    if "open vs code" in command or "open visual studio code" in command:
        speak("Opening VS Code")
        os.system("code")
        return

    if "volume up" in command:
        speak("Increasing volume")
        pyautogui.press("volumeup")
        return

    if "volume down" in command:
        speak("Decreasing volume")
        pyautogui.press("volumedown")
        return

    if "play" in command or "pause" in command:
        speak("Playing or pausing media")
        pyautogui.press("playpause")
        return

    # Gemini handles all other smart commands

    ai_response = ask_gemini(command)

    print("Gemini:", ai_response)

    if ai_response.startswith("OPEN_URL:"):
        url = ai_response.replace("OPEN_URL:", "").strip()

        if not url.startswith("http"):
            url = "https://" + url

        speak("Opening website")
        webbrowser.open(url)
        return

    if ai_response.startswith("SEARCH:"):
        query = ai_response.replace("SEARCH:", "").strip()

        speak(f"Searching for {query}")
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return

    speak(ai_response)

# =========================
# MEDIAPIPE SETUP
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# CAMERA SETUP
# =========================

cap = cv2.VideoCapture(0)

last_action_time = 0
ready_for_palm = True

# =========================
# FINGER DETECTION
# =========================

def count_fingers(hand_landmarks):

    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index, middle, ring, pinky
    for tip in [8, 12, 16, 20]:

        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)

        else:
            fingers.append(0)

    return fingers

# =========================
# GESTURE RECOGNITION
# =========================

def recognize_gesture(fingers):

    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"

    elif fingers == [1, 1, 1, 1, 1]:
        return "Palm"

    elif fingers == [0, 1, 1, 0, 0]:
        return "Victory"

    elif fingers == [0, 1, 0, 0, 0]:
        return "One Finger"

    elif fingers == [0, 1, 1, 1, 0]:
        return "Three Fingers"

    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"

    return "Unknown"

# =========================
# GESTURE ACTIONS
# =========================

def perform_gesture_action(gesture):

    if gesture == "Palm":
        command = listen_command()
        execute_voice_command(command)

    elif gesture == "Victory":
        speak("Opening VS Code")
        os.system("code")

    elif gesture == "One Finger":
        speak("Volume Up")
        pyautogui.press("volumeup")

    elif gesture == "Three Fingers":
        speak("Volume Down")
        pyautogui.press("volumedown")

# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb_frame)

    gesture = "No Hand"

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = count_fingers(hand_landmarks)

            gesture = recognize_gesture(fingers)

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            current_time = time.time()

            if gesture == "Fist":
                speak("Goodbye")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            # Palm activates voice assistant once
            if gesture == "Palm" and ready_for_palm:

                if current_time - last_action_time > 2:

                    perform_gesture_action(gesture)

                    last_action_time = current_time

                    ready_for_palm = False

            # Other gesture shortcuts
            elif gesture in ["Victory", "One Finger", "Three Fingers"]:

                if current_time - last_action_time > 4:

                    perform_gesture_action(gesture)

                    last_action_time = current_time

    else:
        ready_for_palm = True

    cv2.imshow(
        "GestureAI - Gemini AI Assistant",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()