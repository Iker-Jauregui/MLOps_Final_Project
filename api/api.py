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
    description="API to predict track revenue based on reproduction quantity and track features",
    version="2.0.0",
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
    isrc: str = Form(None),
    continent: str = Form(None),
    zone: str = Form(None)
):
    """
    Predict the revenue based on reproduction quantity and track features.

    Parameters
    ----------
    quantity : float
        Number of reproductions/streams/plays for the track.
        Must be non-negative.
    isrc : str, optional
        International Standard Recording Code (ISRC) of the track.
        If not provided or empty, defaults to "UNKNOWN".
    continent : str, optional
        Continent where the track is being played.
        Valid values: "Africa", "Asia", "Europe", "LATAM", "North America", "Oceania".
        If not provided or empty, defaults to "Europe".
    zone : str, optional
        Geographic zone within the continent.
        If not provided or empty, defaults to "UNKNOWN".

    Returns
    -------
    dict
        Dictionary with predicted revenue and input features used.
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
        Response: {
            "quantity": 1000, 
            "isrc": "UNKNOWN", 
            "continent": "Europe", 
            "zone": "UNKNOWN", 
            "predicted_revenue": 2.05
        }
    
    Prediction with custom features:
        POST /predict
        Form data: quantity=1000, isrc="USRC17607839", continent="North America", zone="USA"
        Response: {
            "quantity": 1000, 
            "isrc": "USRC17607839", 
            "continent": "North America", 
            "zone": "USA", 
            "predicted_revenue": 2.15
        }
    """
    # Validate quantity
    if quantity < 0:
        raise HTTPException(
            status_code=400, 
            detail="'quantity' must be a non-negative value"
        )

    try:
        # Handle empty strings and None values for optional parameters
        # Empty strings from form data should be treated as None
        isrc_value = isrc if isrc and isrc.strip() else None
        continent_value = continent if continent and continent.strip() else None
        zone_value = zone if zone and zone.strip() else None
        
        # Get prediction - logic.regressor will handle encoding
        revenue = predict_func(
            quantity=quantity,
            isrc=isrc_value,
            continent=continent_value,
            zone=zone_value
        )

        # Return prediction with all feature values used (for transparency)
        # Show what the model actually used (after defaults applied in logic.regressor)
        return {
            "quantity": quantity,
            "isrc": isrc_value if isrc_value is not None else "UNKNOWN",
            "continent": continent_value if continent_value is not None else "Europe",
            "zone": zone_value if zone_value is not None else "UNKNOWN",
            "predicted_revenue": revenue
        }

    except ValueError as e:
        # Handle validation errors from logic.regressor (e.g., invalid continent)
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid input: {str(e)}"
        ) from e
    except (TypeError, KeyError) as e:
        # Handle other processing errors
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing prediction: {str(e)}"
        ) from e
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        ) from e


# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
