import cv2 as cv
import joblib
import numpy as np
import json
from wavelet import w2d
import base64

__model = None
__emotion_classes = None

def list_emotion_classes():
    return list(__emotion_classes.keys())

def predict_emotion(image):
    try:
        input = img_transformation(image)
        pred_num = __model.predict([input])[0]
        emotion = list(__emotion_classes.keys())[pred_num]
        print(emotion)
        return emotion
    except:
        return "Faces not detected"

def get_cropped_face(img_blur):
    face_cascade = cv.CascadeClassifier("face_detection/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img_blur)
    # print(faces)
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            new_img = img_blur[y:y+h, x:x+w]
            return new_img
    else:
        return None

def img_transformation(image):
    img = cv.GaussianBlur(image, (3, 3), 0)
    cropped_img = get_cropped_face(img)
    # if cropped_img:
    scaled_processed_img = cv.resize(cropped_img, (32, 32))
    img_har = w2d(cropped_img, 'db1', 5)
    scaled_raw_har = cv.resize(img_har, (32, 32))
    combined_img = np.vstack((scaled_processed_img.reshape(32 * 32 * 3, 1), scaled_raw_har.reshape(32 * 32, 1)))
    input_img = combined_img.reshape(4096).astype(float)
    print(input_img)
    return input_img

def get_cv2_image_from_base64_string(img_base64_data):
    encoded_data = img_base64_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return img

def extract_base64_from_file():
    with open("b64.txt", "r") as f:
        return f.read()

def load_artifacts():
    print("Start loading artifacts.........!!!!")
    global __model
    global __emotion_classes

    with open("artifacts/emotion_classes.json","r") as f:
        __emotion_classes = json.load(f)

    with open("artifacts/updated_emotion_model.pkl","rb") as f:
        __model = joblib.load(f)

    print("Successfully loaded artifacts.........!!!")

if __name__ == "__main__":
    load_artifacts()
    img = get_cv2_image_from_base64_string(extract_base64_from_file())
    predict_emotion(img)
    # print(list_emotion_classes())
