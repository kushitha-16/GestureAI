import cv2
import mediapipe as mp
import joblib
import pandas as pd

MODEL_FILE = "gesture_model.pkl"

model = joblib.load(MODEL_FILE)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)


def extract_landmarks(hand_landmarks):
    data = []

    for landmark in hand_landmarks.landmark:
        data.extend([landmark.x, landmark.y, landmark.z])

    return data


while True:
    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    predicted_gesture = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = extract_landmarks(hand_landmarks)

            prediction = model.predict([landmarks])[0]

            predicted_gesture = prediction

            cv2.putText(
                frame,
                f"ML Gesture: {predicted_gesture}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    else:
        cv2.putText(
            frame,
            "No Hand",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("GestureAI ML Gesture Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()