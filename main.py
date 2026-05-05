from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io

from model import extract_image_features
from utils import generate_response

app = FastAPI()

@app.get("/")
def home():
    return {"message": "VisionFusion AI API Running"}

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...), prompt: str = Form(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    features = extract_image_features(image)
    result = generate_response(prompt, features)

    return {
        "query": prompt,
        "result": result
    }