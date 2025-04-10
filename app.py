import os
import shutil
import uuid
from datetime import date
from typing import Optional

import pandas as pd
import requests
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from huggingface_hub import InferenceClient
from pydantic import BaseModel

from utils.detector import process_video
from utils.predict import predict_sales_quantity
from utils.tainer import train_model

app = FastAPI()

HF_API_TOKEN = os.getenv("HF_TOKEN")
TEXT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
client = InferenceClient(token=HF_API_TOKEN)
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


class GenerateRequest(BaseModel):
    prompt: str


class AccuracyMetrics(BaseModel):
    r2_score: Optional[float]
    mean_absolute_error: Optional[float]


class PredictResponse(BaseModel):
    predicted_quantity_sold: int
    model_accuracy: AccuracyMetrics


class PredictRequest(BaseModel):
    date: date
    color: str


@app.post("/generate")
def generate(req: GenerateRequest):
    # Instruction prompt to DeepSeek to decide what to do
    system_prompt = (
        "Determine the user's intent from the prompt below. "
        "Possible actions are: summarize, translate, generate code, or generate an image. "
        "Do not output the action name—just perform the task. "
        "If the task is image generation, prepend the user prompt with 'Image: ' and keep the original user prompt. "
        "Only return the result of the task—no explanations, no extra text, no instructions. "
        "Respond with the direct output of the task only. "
        f"User prompt: {req.prompt}\n"
    )

    try:
        # request to AI model
        completion = client.chat_completion(
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.7,
            max_tokens=250,
            model=TEXT_MODEL,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if text.startswith("Image:"):
        # Send result (image description) to Stable Diffusion
        payload = {"inputs": text.strip()}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}",
            headers=HEADERS,
            json=payload,
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        os.makedirs("assets/images", exist_ok=True)

        # Save image to file
        image_path = f"assets/images/{uuid.uuid4()}.png"
        with open(image_path, "wb") as f:
            f.write(response.content)

        return FileResponse(
            image_path,
            media_type="image/png",
            status_code=200,
        )

    return JSONResponse(
        content={"task": "text_response", "response": text},
        media_type="application/json",
        status_code=200,
    )


@app.post("/train_model")
async def train(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if not {"date", "color", "quantity_sold"}.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'date', 'color', and 'quantity_sold' columns",
            )

        # run training in the background
        BackgroundTasks().add_task(train_model, df)
        return JSONResponse({"message": "Training started"}, status_code=202)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        prediction = predict_sales_quantity(request.date, request.color)
        return PredictResponse(**prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/detect-cars")
async def detect_cars(
    video: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    return_video: bool = Form(False),
):
    # Save uploaded video
    input_path = f"assets/videos/{uuid.uuid4()}.mp4"
    os.makedirs("assets/videos", exist_ok=True)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Run detection and counting
    result = process_video(input_path, (x1, y1), (x2, y2), return_video)

    if return_video:
        return FileResponse(
            result["output_path"], media_type="video/mp4", filename="output.mp4"
        )
    else:
        return JSONResponse({"cars_crossed": result["count"]})
