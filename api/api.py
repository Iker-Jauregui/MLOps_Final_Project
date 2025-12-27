"""FastAPI application for track revenue prediction."""

import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from lib.regressor import predict as predict_func

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
):
    """
    Predict the revenue based on reproduction quantity.

    Parameters
    ----------
    quantity : float
        Number of reproductions/streams/plays for the track.
        Must be non-negative.

    Returns
    -------
    dict
        Dictionary with predicted revenue and input quantity.
        Contains keys: 'quantity', 'predicted_revenue'

    Raises
    ------
    HTTPException
        If quantity is negative or if an error occurs during prediction.
    """
    if quantity < 0:
        raise HTTPException(
            status_code=400, detail="'quantity' must be a non-negative value"
        )

    try:
        # Get prediction
        revenue = predict_func(quantity)

        return {
            "quantity": quantity,
            "predicted_revenue": revenue
        }

    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing prediction: {str(e)}"
        ) from e


# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
