import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATA_FILE = "gesture_data.csv"

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# 21 landmarks × x,y,z = 63 features
def extract_landmarks(hand_landmarks):
    data = []

    for landmark in hand_landmarks.landmark:
        data.extend([landmark.x, landmark.y, landmark.z])

    return data


# Create CSV header if file does not exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)

        header = []

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        header.append("label")
        writer.writerow(header)


print("Gesture Data Collection Started")
print("--------------------------------")
print("Press keys to save samples:")
print("p = palm")
print("f = fist")
print("v = victory")
print("o = one_finger")
print("t = three_fingers")
print("u = thumbs_up")
print("k = ok_sign")
print("r = rock")
print("c = call_me")
print("s = stop")
print("e = peace")
print("q = quit")
print("--------------------------------")

while True:
    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    current_landmarks = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            current_landmarks = extract_landmarks(hand_landmarks)

            cv2.putText(
                frame,
                "Hand Detected - Press key to save sample",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    else:
        cv2.putText(
            frame,
            "No Hand Detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Collect Gesture Data", frame)

    key = cv2.waitKey(1) & 0xFF

    label = None

    if key == ord("p"):
        label = "palm"

    elif key == ord("f"):
        label = "fist"

    elif key == ord("v"):
        label = "victory"

    elif key == ord("o"):
        label = "one_finger"

    elif key == ord("t"):
        label = "three_fingers"

    elif key == ord("u"):
        label = "thumbs_up"
    
    elif key == ord("k"):
         label = "ok_sign"
    elif key == ord("r"):
         label = "rock"
    elif key == ord("c"):
         label = "call_me"
    elif key == ord("s"):
         label = "stop"
    elif key == ord("e"):
         label = "peace"

    elif key == ord("q"):
        break

    if label and current_landmarks:
        with open(DATA_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(current_landmarks + [label])

        print(f"Saved sample: {label}")

cap.release()
cv2.destroyAllWindows()