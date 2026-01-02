import gradio as gr
import requests

# URL of the API created with FastAPI
API_URL = "https://mlops-final-project-latest-tu7z.onrender.com"

# Continent options for dropdown
CONTINENT_OPTIONS = [
    "Africa",
    "Asia",
    "Europe",
    "LATAM",
    "North America",
    "Oceania"
]


# Function to execute when clicking the "Predict" button
def predict(isrc, continent, zone, quantity):
    """
    Send track features to the FastAPI endpoint and return predicted revenue.
    
    Parameters
    ----------
    isrc : str
        International Standard Recording Code (can be empty)
    continent : str
        Continent where the track is played (selected from dropdown)
    zone : str
        Geographic zone (can be empty)
    quantity : float
        Number of track reproductions/streams/plays
        
    Returns
    -------
    str
        Formatted prediction result or error message
    """
    try:
        # Validate quantity input
        if quantity is None:
            return "❌ **Error:** Please enter a quantity value"
        
        if quantity < 0:
            return "❌ **Error:** Quantity must be non-negative"
        
        # Validate continent
        if not continent:
            return "❌ **Error:** Please select a continent"
        
        # Prepare request payload
        payload = {
            "quantity": str(quantity),
            "continent": continent
        }
        
        # Add optional fields only if provided
        if isrc and isrc.strip():
            payload["isrc"] = isrc.strip()
        
        if zone and zone.strip():
            payload["zone"] = zone.strip()
        
        # Send POST request to the API
        response = requests.post(
            f"{API_URL}/predict",
            data=payload,
            timeout=120
        )
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        predicted_revenue = data.get("predicted_revenue")
        
        # Extract input features from response (API will return what it used)
        input_isrc = data.get("isrc", "Not provided")
        input_continent = data.get("continent", continent)
        input_zone = data.get("zone", "Not provided")
        input_quantity = data.get("quantity", quantity)
        
        # Format the result
        result = f"""
### 🎯 Prediction Results

**Input Features:**
- **ISRC:** {input_isrc}
- **Continent:** {input_continent}
- **Zone:** {input_zone}
- **Quantity:** {input_quantity:,.0f
