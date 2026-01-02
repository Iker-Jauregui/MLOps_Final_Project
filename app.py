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
- **Quantity:** {input_quantity:,.0f} plays

---

**💰 Predicted Revenue:** {predicted_revenue:,.2f}€
        """
        return result.strip()
        
    except requests.exceptions.ConnectionError:
        return f"❌ **Error:** Could not connect to API at {API_URL}.\n\nPlease ensure the server is running."
    except requests.exceptions.Timeout:
        return "❌ **Error:** Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = e.response.json().get("detail", e.response.text)
        except:
            error_detail = e.response.text
        return f"❌ **Error:** API returned status code {e.response.status_code}\n\n{error_detail}"
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


# GUI created using Gradio
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="ISRC (International Standard Recording Code)",
            placeholder="e.g., USRC17607839 (optional)",
            value="",
            info="Leave empty if unknown - the model will use a default value"
        ),
        gr.Dropdown(
            label="Continent",
            choices=CONTINENT_OPTIONS,
            value="Europe",
            info="Select the continent where the track is being played"
        ),
        gr.Textbox(
            label="Zone",
            placeholder="e.g., Western Europe, East Asia (optional)",
            value="",
            info="Leave empty if unknown - the model will use a default value"
        ),
        gr.Number(
            label="Number of Reproductions/Plays",
            value=1000,
            minimum=0,
            step=1,
            info="Enter the number of times the track has been played"
        ),
    ],
    outputs=gr.Markdown(label="Prediction Result"),
    title="🎵 Track Revenue Predictor",
    description="""
    **Predict track revenue based on multiple features**
    
    This application uses a machine learning model trained on real streaming data to predict revenue.
    
    **How to use:**
    1. (Optional) Enter the ISRC code of the track
    2. Select the continent where the track is being played
    3. (Optional) Enter the specific zone within the continent
    4. Enter the number of reproductions/streams/plays
    5. Click **Submit** to get the predicted revenue
    
    **Note:** ISRC and Zone fields are optional. If left empty, the model will use appropriate default values.
    """,
    examples=[
        ["USRC17607839", "Europe", "Western Europe", 1000],
        ["", "North America", "", 5000],
        ["GBUM71507547", "Asia", "East Asia", 10000],
        ["", "LATAM", "South America", 50000],
        ["USUM71808193", "Oceania", "", 100000],
        ["", "Africa", "North Africa", 500000],
    ],
    flagging_options=None,
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 800px !important;
    }
    """
)

# Launch the GUI
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False  # Set to True if you want a public link
    )
