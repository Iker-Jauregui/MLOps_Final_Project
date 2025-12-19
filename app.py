import gradio as gr
import requests
import numpy as np
import cv2

# URL of the API created with FastAPI
API_URL = "https://mlops-final-project-latest-tu7z.onrender.com"


# Function to execute when clicking the "Predict button"
def predict(image):
    try:
        _, img_encoded = cv2.imencode(".jpg", image)
        files = {"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}

        response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("predicted_class")
    except Exception as e:
        return f"Error: {str(e)}"


# GUI creted using Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Image", type="numpy", height=400),
    outputs=gr.Textbox(label="Predicted class"),
    title="Image class predictor with FastAPI and Gradio",
    description="Interactive class predictor using the endpoint /predict",
)

# Launch the GUI
if __name__ == "__main__":
    iface.launch()
