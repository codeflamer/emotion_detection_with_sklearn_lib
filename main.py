import util as util
from pydantic import BaseModel

from fastapi import FastAPI

class EmotionParam(BaseModel):
    image:str

app = FastAPI()
util.load_artifacts()

@app.get("/")
async def get_home():
    return {"status": "OK!!"}

@app.get("/emotion_classes")
async def get_classes():
    return {"classes": util.list_emotion_classes()}

@app.post("/detect_emotion")
async def detect_emotion(param:EmotionParam):
    image = param.image
    return {"response": util.predict_emotion(util.get_cv2_image_from_base64_string(image))}
