"""FastAPI application for image classification."""

import io
import uvicorn
from PIL import Image
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from mylib.classifier import predict as predict_func
from mylib.classifier import resize as resize_func

# Create an instance of FastAPI
app = FastAPI(
    title="API of the Image Classifier using FastAPI",
    description="API to perform image predictions and transforms using mylib.classifier",
    version="1.0.0",
)

# We use the templates folder to obtain HTML files
templates = Jinja2Templates(directory="templates")


# Initial endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home endpoint returning HTML template."""
    return templates.TemplateResponse(request=request, name="home.html")


# Main endpoint to perform the image prediction
@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    class_names: str = Form(default="cardboard,paper,plastic,metal,trash,glass"),
):
    """
    Predict the class of the input image.

    Parameters
    ----------
    file : UploadFile
        Image file to classify
    class_names : str
        Comma-separated class names (default: "cardboard,paper,plastic,metal,trash,glass")

    Returns
    -------
    dict
        Dictionary with predicted class
    """
    try:
        # Read image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert class_names string to list
        class_list = [c.strip() for c in class_names.split(",")]

        # Get prediction
        prediction = predict_func(image, class_list)

        return {"predicted_class": prediction}

    except (FileNotFoundError, IOError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing image: {str(e)}"
        ) from e


# Main endpoint to perform the image resize
@app.post("/resize")
async def resize_endpoint(
    file: UploadFile = File(...), width: int = Form(...), height: int = Form(...)
):
    """
    Resize the input image.

    Parameters
    ----------
    file : UploadFile
        Image file to resize
    width : int
        Target width (must be positive)
    height : int
        Target height (must be positive)

    Returns
    -------
    dict
        Dictionary with new image dimensions
    """
    if width <= 0:
        raise HTTPException(status_code=400, detail="'width' must be a positive value")
    if height <= 0:
        raise HTTPException(status_code=400, detail="'height' must be a positive value")

    try:
        # Read image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Resize image
        new_size = resize_func(image, width, height)

        return {"resized_dimensions": new_size}

    except (FileNotFoundError, IOError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"Error resizing image: {str(e)}"
        ) from e


# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
