# predict.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import argparse
import cv2

# --- Configuration ---
MODEL_PATH = "./models/ProposedNetwork_StratifiedKFold_Final_fold_5.hdf5" # Path to your best model
IMAGE_SIZE = 224

def getLabel(class_id):
    """Returns the string label for a given class ID."""
    return 'NoFire' if class_id == 0 else 'Fire'

# --- Argument Parser ---
# This allows us to run the script from the command line
parser = argparse.ArgumentParser(description="Predict if an image contains fire using a trained model.")
parser.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(parser.parse_args())

# --- Load and Preprocess the Image ---
try:
    print("[INFO] Loading and preprocessing image...")
    # Load the image using Keras helper utility
    image = load_img(args["image"], target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = img_to_array(image)
    
    # Scale pixel values to the range [0, 1] as done during training
    image = image / 255.0
    
    # Add a batch dimension, as the model expects it
    image = np.expand_dims(image, axis=0)
except Exception as e:
    print("[ERROR] Could not load or process the image. Please check the path.")
    print("Error details:", e)
    exit()


# --- Load the Model and Predict ---
try:
    print("[INFO] Loading the trained model...")
    model = load_model(MODEL_PATH)

    print("[INFO] Making prediction...")
    # Get the raw prediction (probabilities)
    prediction_probs = model.predict(image)[0]
    
    # Get the class index with the highest probability
    predicted_class_index = np.argmax(prediction_probs)
    
    # Get the corresponding label and confidence
    predicted_label = getLabel(predicted_class_index)
    confidence = prediction_probs[predicted_class_index]

    # --- Display the Result ---
    print("\n========== PREDICTION RESULT ==========")
    print("Prediction: {}".format(predicted_label))
    print("Confidence: {:.2f}%".format(confidence * 100))
    print("=======================================")

    # Optional: Display the image with the prediction
    original_image = cv2.imread(args["image"])
    text = "{}: {:.2f}%".format(predicted_label, confidence * 100)
    color = (0, 0, 255) if predicted_label == "Fire" else (0, 255, 0)
    cv2.putText(original_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Result", original_image)
    cv2.waitKey(0) # Press any key to close the image window
    cv2.destroyAllWindows()

except Exception as e:
    print("[ERROR] An error occurred during model loading or prediction.")
    print("Error details:", e)