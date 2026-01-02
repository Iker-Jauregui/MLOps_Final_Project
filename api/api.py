"""FastAPI application for track revenue prediction."""

import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from logic.regressor import predict as predict_func

# Create an instance of FastAPI
app = FastAPI(
    title="API of the Track Revenue Predictor using FastAPI",
    description="API to predict track revenue based on reproduction quantity",
    version="1.0.0",
)

# We use the templates folder to obtain HTML files
templates = Jinja2Templates(directory="templates")


# Initial endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home endpoint returning HTML template."""
    return templates.TemplateResponse(request=request, name="home.html")


# Main endpoint to perform the revenue prediction
@app.post("/predict")
async def predict_endpoint(
    quantity: float = Form(...),
    isrc: int = Form(None),
    continent: int = Form(None),
    zone: int = Form(None)
):
    """
    Predict the revenue based on reproduction quantity and optional categorical features.

    Parameters
    ----------
    quantity : float
        Number of reproductions/streams/plays for the track.
        Must be non-negative.
    isrc : int, optional
        ISRC code (encoded as integer). Defaults to 0 (UNKNOWN).
    continent : int, optional
        Continent code (encoded as integer). Defaults to 3 (Europe).
    zone : int, optional
        Zone code (encoded as integer). Defaults to 0 (UNKNOWN).

    Returns
    -------
    dict
        Dictionary with predicted revenue, input quantity, and feature values used.
        Contains keys: 'quantity', 'isrc', 'continent', 'zone', 'predicted_revenue'

    Raises
    ------
    HTTPException
        If quantity is negative or if an error occurs during prediction.
    
    Examples
    --------
    Basic prediction with defaults:
        POST /predict
        Form data: quantity=1000
        Response: {"quantity": 1000, "isrc": 0, "continent": 3, "zone": 0, "predicted_revenue": 2.05}
    
    Prediction with custom features:
        POST /predict
        Form data: quantity=1000, isrc=42, continent=1, zone=5
        Response: {"quantity": 1000, "isrc": 42, "continent": 1, "zone": 5, "predicted_revenue": 2.15}
    """
    if quantity < 0:
        raise HTTPException(
            status_code=400, detail="'quantity' must be a non-negative value"
        )

    try:
        # Get prediction with optional categorical features
        revenue = predict_func(
            quantity=quantity,
            isrc=isrc,
            continent=continent,
            zone=zone
        )

        # Return prediction with all feature values used (for transparency)
        return {
            "quantity": quantity,
            "isrc": isrc if isrc is not None else 0,      # Show actual default used
            "continent": continent if continent is not None else 3,
            "zone": zone if zone is not None else 0,
            "predicted_revenue": revenue
        }

    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing prediction: {str(e)}"
        ) from e


# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
