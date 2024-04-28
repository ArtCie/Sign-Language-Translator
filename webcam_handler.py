import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

while webcam.isOpened():
    success, img = webcam.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = mp_hands.Hands().process(img_rgb)

    if results.multi_hand_landmarks:
        print("Results:")
        for hand_landmarks in results.multi_hand_landmarks:
            print("NEW landmark!")
            for point in hand_landmarks.landmark:
                print(point)
                mp_drawing.draw_landmarks(img, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Test', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
