import gradio as gr
import requests

# URL of the API created with FastAPI
API_URL = "https://mlops-final-project-latest-tu7z.onrender.com"


# Function to execute when clicking the "Predict" button
def predict(quantity):
    """
    Send quantity to the FastAPI endpoint and return predicted revenue.
    
    Parameters
    ----------
    quantity : float
        Number of track reproductions/streams/plays
        
    Returns
    -------
    str
        Formatted prediction result or error message
    """
    try:
        # Validate input
        if quantity is None:
            return "Error: Please enter a quantity value"
        
        if quantity < 0:
            return "Error: Quantity must be non-negative"
        
        # Send POST request to the API
        response = requests.post(
            f"{API_URL}/predict",
            data={"quantity": str(quantity)},
            timeout=120
        )
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        predicted_revenue = data.get("predicted_revenue")
        input_quantity = data.get("quantity")
        
        # Format the result
        result = f"""
### Prediction Results

**Input Quantity:** {input_quantity:,.0f} plays

**Predicted Revenue:** {predicted_revenue:,.2f}€

**Rate:** 2.00€ per 1,000 plays
        """
        return result.strip()
        
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to API at {API_URL}. Please ensure the server is running."
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"Error: API returned status code {e.response.status_code}. {e.response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


# GUI created using Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Number(
        label="Number of Reproductions/Plays",
        value=1000,
        minimum=0,
        step=1,
        info="Enter the number of times the track has been played"
    ),
    outputs=gr.Markdown(label="Prediction Result"),
    title="🎵 Track Revenue Predictor",
    description="""
    **Predict track revenue based on reproduction quantity**
    
    This application uses a FastAPI backend to predict revenue using the formula:
    
    Revenue = 2.0 × Quantity / 1000
    
    Enter the number of reproductions/streams/plays and click **Submit** to get the predicted revenue.
    """,
    examples=[
        [1000],
        [5000],
        [10000],
        [50000],
        [100000],
        [500000],
    ],
    flagging_options=None,
)

# Launch the GUI
if __name__ == "__main__":
    iface.launch()