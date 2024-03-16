import cv2
import numpy as np
from keras.models import model_from_json, load_model
import pandas as pd
import mediapipe as mp
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load emotion model


def load_emotion_model():
    json_file = open('emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("emotion_model.h5")
    # print("Loaded emotion model from disk")
    return emotion_model

# emotion_model_without_aug_v2

# Load gesture model


def load_gesture_model():
    return load_model('model_10epoch.h5')


# Use camera 1 (change to 0 if using the default camera)
cap = cv2.VideoCapture(1)

emotion_model = load_emotion_model()
gesture_model = load_gesture_model()

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

ret, frame = cap.read()
h, w, c = frame.shape
flip_state = False

analysisframe = ''
gesturePred = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace",
               "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]

while True:
    _, frame = cap.read()
    if not ret:
        break

    face_detector = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x_face, y_face, w_face, h_face) in num_faces:
        cv2.rectangle(frame, (x_face, y_face-50),
                      (x_face+w_face, y_face+h_face+10), (14, 172, 20), 4)
        roi_gray_frame = gray_frame[y_face:y_face +
                                    h_face, x_face:x_face + w_face]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        maxconfidence = round(np.max(emotion_prediction[0]) * 100, 2)

        emotion_text = f'{emotion_dict[maxindex]} {maxconfidence}%'
        cv2.putText(frame, emotion_text, (x_face+5, y_face-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                # Ensure x is within the frame boundaries
                x = max(0, min(x, w-1))
                # Ensure y is within the frame boundaries
                y = max(0, min(y, h-1))
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (14, 172, 20), 2)

            analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            analysisframe = cv2.resize(analysisframe, (48, 48))

            nlist = []
            rows, cols = analysisframe.shape
            for i in range(rows):
                for j in range(cols):
                    k = analysisframe[i, j]
                    nlist.append(k)

            datan = pd.DataFrame(nlist).T
            colname = []
            for val in range(2304):
                colname.append(val)
            datan.columns = colname

            pixeldata = datan.values
            pixeldata = pixeldata / 255
            pixeldata = pixeldata.reshape(-1, 48, 48, 1)
            prediction = gesture_model.predict(pixeldata)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, gesturePred[maxindex], (x+5, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            predarray = np.array(prediction[0])
            letter_prediction_dict = {
                gesturePred[i]: predarray[i] for i in range(len(gesturePred))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]
            high2 = predarrayordered[1]
            high3 = predarrayordered[2]
            # for key, value in letter_prediction_dict.items():
            #     if value == high1:
            #         print("Predicted Gesture 1: ", key)
            #         print('Confidence 1: ', 100*value)
            #     elif value == high2:
            #         print("Predicted Gesture 2: ", key)
            #         print('Confidence 2: ', 100*value)
            #     elif value == high3:
            #         print("Predicted Gesture 3: ", key)
            #         print('Confidence 3: ', 100*value)

    if flip_state:
        frame = cv2.flip(frame, 1)

    cv2.imshow('Combined Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
