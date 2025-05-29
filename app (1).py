### `app.py`

```python
import numpy as np
import gradio as gr
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib # To save and load the trained model and scaler

# --- Model Loading (assuming models are trained and saved) ---
# In a real scenario, you would train these models in a separate script/notebook
# and save them. For this app.py, we'll quickly train them if not found,
# or load them if they exist.

MODEL_PATH = 'trained_random_forest_model.joblib' # Specific name for clarity
SCALER_PATH = 'trained_scaler.joblib' # Scaler is mostly for SGD/KNN, but included for completeness

# Function to train and save models (if they don't exist)
def train_and_save_models():
    print("Loading MNIST dataset for app.py (full dataset for final model)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target
    y = y.astype(np.uint8)
    
    # Normalize pixel values to be between 0 and 1
    X_normalized = X / 255.0

    print("Training RandomForestClassifier for app.py...")
    # Train RF on the 0-1 normalized data, as it's less sensitive to StandardScaler
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    rf_clf.fit(X_normalized, y)
    print("RandomForestClassifier training complete for app.py.")

    # A StandardScaler is fitted here, but for the RF in the app, 0-1 normalization is usually sufficient.
    # It would be critical if the app were to use a StandardScaler-sensitive model like SGD or KNN.
    scaler = StandardScaler()
    scaler.fit(X.astype(np.float64)) # Fit scaler on original unnormalized X for consistency if used elsewhere

    # Save the trained model and scaler
    joblib.dump(rf_clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH) # Save scaler even if not used by RF in app, for potential future use
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    return rf_clf, scaler

# Load models or train if not found
try:
    rf_clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) # Load scaler, though not strictly used by RF in app
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Scaler loaded from {SCALER_PATH}")
except FileNotFoundError:
    print("Model or scaler not found. Training and saving new models...")
    rf_clf, scaler = train_and_save_models()

def predict_digit_gradio(input_data):
    """
    Handles sketchpad input from Gradio, preprocesses it, and makes predictions
    using the trained Random Forest Classifier.

    Args:
        input_data: The input from Gradio's gr.Sketchpad.
                    This is typically a dictionary {'background': ..., 'layers': ..., 'composite': numpy_array}
                    or a numpy_array directly for older Gradio versions.

    Returns:
        A dictionary formatted for gr.Label, containing predicted class probabilities.
        Returns a "No drawing detected" or "Error" message with 0.0 confidence if issues occur.
    """
    image = None # Initialize image variable

    # Robustly extract the image data from input_data
    if isinstance(input_data, dict):
        if 'composite' in input_data and input_data['composite'] is not None:
            image = input_data['composite']
        elif 'image' in input_data and input_data['image'] is not None:
            image = input_data['image']
        else:
            return {"No drawing detected": 0.0}
    elif isinstance(input_data, np.ndarray):
        image = input_data
    else:
        return {"Error: Invalid input format": 0.0}

    # Ensure the image is a NumPy array and not empty
    if image is None or (isinstance(image, np.ndarray) and image.size == 0):
        return {"No drawing detected": 0.0}

    image = np.asarray(image)
    
    # Convert to grayscale if needed (Sketchpad often returns RGBA)
    if image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3].mean(axis=2) # Take mean of RGB channels
    
    # Define target size for MNIST (28x28)
    target_size = (28, 28) 
    
    # Resize the image to 28x28
    if image.shape != target_size:
        h, w = image.shape
        if h == 0 or w == 0:
            return {"No drawing detected": 0.0}

        scale_x = w / target_size[1]
        scale_y = h / target_size[0]
        
        j, i = np.meshgrid(
            np.clip((np.arange(target_size[1]) * scale_x).astype(int), 0, w - 1),
            np.clip((np.arange(target_size[0]) * scale_y).astype(int), 0, h - 1)
        )
        image = image[i, j]

    # Invert colors: Sketchpad draws black on white, but MNIST models expect white on black
    image = 255 - image 
    
    # Normalize pixel values to be between 0 and 1
    image_flat = image.reshape(1, -1)
    if np.max(image_flat) > 0:
        image_normalized = image_flat / np.max(image_flat) # Normalize based on actual max
    else:
        return {"No drawing detected": 0.0}

    # Check if the drawing is too faint after normalization (optional, but good for robustness)
    if np.sum(image_normalized) < 5: # Threshold can be tuned
        return {"No drawing detected": 0.0}

    # Get predictions from the Random Forest Classifier
    try:
        # The Random Forest model was trained on 0-1 scaled data (X_normalized from train_and_save_models).
        # The input image here (image_normalized) is also 0-1 scaled, so it's compatible.
        rf_proba = rf_clf.predict_proba(image_normalized)[0]
        rf_results = {str(i): prob for i, prob in enumerate(rf_proba)}
        
        return rf_results

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"Prediction Error": 0.0}

# Create Gradio interface
interface = gr.Interface(
    fn=predict_digit_gradio,
    inputs=gr.Sketchpad(image_mode="L"), # Removed brush_radius
    outputs="label",
    title="MNIST Digit Recognition",
    description="Draw a digit (0-9) on the white canvas with a black brush. The model is trained on the MNIST dataset (28x28 pixels).",
    live=True
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
