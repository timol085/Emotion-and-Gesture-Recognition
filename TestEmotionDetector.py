import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def load_emotion_model():
    json_file = open('emotion_model_without_aug.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("emotion_model_without_aug.h5")
    print("Loaded model from disk")
    return emotion_model

cap = cv2.VideoCapture(1)

emotion_model = load_emotion_model()

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (14, 172, 20), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        maxconfidence = round(np.max(emotion_prediction[0]) * 100, 2)

        # Draw emotion text with confidence
        emotion_text = f'{emotion_dict[maxindex]} {maxconfidence}%'
        cv2.putText(frame, emotion_text, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# from keras.models import model_from_json
# import mediapipe as mp

# # Load emotion detection model
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# def load_emotion_model():
#     json_file = open('emotion_model_without_aug.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     emotion_model = model_from_json(loaded_model_json)
#     emotion_model.load_weights("emotion_model_without_aug.h5")
#     print("Loaded emotion model from disk")
#     return emotion_model

# # Initialize video capture
# cap = cv2.VideoCapture(1)

# # Initialize Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3, min_detection_confidence=0.5) as face_mesh:
#     emotion_model = load_emotion_model()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Emotion detection
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#         num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#         for (x, y, w, h) in num_faces:
#             roi_gray_frame = gray_frame[y:y + h, x:x + w]
#             cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#             emotion_prediction = emotion_model.predict(cropped_img)
#             maxindex = int(np.argmax(emotion_prediction))
#             maxconfidence = round(np.max(emotion_prediction[0]) * 100, 2)   

#             emotion_text = f'{emotion_dict[maxindex]} {maxconfidence}%'
#             cv2.putText(frame, emotion_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

#         # Face mesh detection
#         results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image=frame,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_TESSELATION,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
#                 )

#         cv2.imshow('Face Mesh and Emotion Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

