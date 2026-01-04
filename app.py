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
        result = f"""### 🎯 Prediction Results

**Input Features:**
- **ISRC:** {input_isrc}
- **Continent:** {input_continent}
- **Zone:** {input_zone}
- **Quantity:** {input_quantity:,.0f} plays

---

**💰 Predicted Revenue:** {predicted_revenue:,.2f}€"""
        
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
with gr.Blocks(title="🎵 Track Revenue Predictor") as iface:
    gr.Markdown("# 🎵 Track Revenue Predictor")
    gr.Markdown("""
    **Predict track revenue based on multiple features**
    
    This application uses a machine learning model trained on real streaming data to predict revenue.
    
    **How to use:**
    1. (Optional) Enter the ISRC code of the track
    2. Select the continent where the track is being played
    3. (Optional) Enter the specific zone within the continent
    4. Enter the number of reproductions/streams/plays
    5. Click **Submit** to get the predicted revenue
    
    **Note:** ISRC and Zone fields are optional. If left empty, the model will use appropriate default values.
    """)
    
    with gr.Row():
        with gr.Column():
            isrc_input = gr.Textbox(
                label="ISRC (International Standard Recording Code)",
                placeholder="e.g., FR-X87-24-72070 (optional)",
                value="",
            )
            continent_input = gr.Dropdown(
                label="Continent",
                choices=CONTINENT_OPTIONS,
                value="Europe",
            )
            zone_input = gr.Textbox(
                label="Zone",
                placeholder="e.g., Spain (optional)",
                value="",
            )
            quantity_input = gr.Number(
                label="Number of Reproductions/Plays",
                value=1000,
                minimum=0,
            )
            
            submit_btn = gr.Button("Submit", variant="primary")
        
        with gr.Column():
            output = gr.Markdown(label="Prediction Result")
    
    # Examples section
    gr.Markdown("### 📝 Example Inputs")
    gr.Examples(
        examples=[
            ["FR-X87-24-72070", "Europe", "France", 1000],
            ["", "North America", "", 5000],
            ["GB-SXS-24-00075", "Asia", "Japan", 10000],
            ["", "LATAM", "Mexico", 50000],
            ["UK-XN2-20-67122", "Oceania", "", 100000],
            ["", "Africa", "Egypt", 500000],
        ],
        inputs=[isrc_input, continent_input, zone_input, quantity_input],
    )
    
    # Connect the button to the prediction function
    submit_btn.click(
        fn=predict,
        inputs=[isrc_input, continent_input, zone_input, quantity_input],
        outputs=output,
    )

# Launch the GUI
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
