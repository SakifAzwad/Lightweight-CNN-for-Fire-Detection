import os
import pandas as pd
import numpy as np
import matplotlib
# Use a backend that doesn't require a GUI, which is more reliable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from imutils import paths
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ==============================================================================
# 1. CONFIGURATION - !!! ONLY EDIT THIS SECTION !!!
# ==============================================================================
FOLD_TO_USE = 5
NUM_IMAGES = 8
MODEL_NAME = "ProposedNetwork_StratifiedKFold_Final"
DATASET_PATH = './Dataset/ThermalFire/'
MODEL_DIR = "./Model"
PLOT_DIR = "./Plot"
N_SPLITS = 5
NORM_SIZE = 224
BATCH_SIZE = 1

# ==============================================================================
# DO NOT EDIT BELOW THIS LINE
# ==============================================================================

print("[INFO] Starting Grad-CAM script...")
os.makedirs(PLOT_DIR, exist_ok=True)
model_path = "{}/{}_fold_{}.hdf5".format(MODEL_DIR, MODEL_NAME, FOLD_TO_USE)

if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found at: {}\nPlease ensure the model for Fold {} exists.".format(model_path, FOLD_TO_USE))

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def getLabel(class_id):
    """Returns the string label for a given class ID."""
    return ['NoFire', 'Fire'][class_id]

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    if grads is None:
        print("[ERROR] Gradient calculation failed. The gradient is None.")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def superimpose_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# ==============================================================================
# 3. LOAD DATA AND MODEL
# ==============================================================================
print("[INFO] Loading model from: {}".format(model_path))
model = load_model(model_path)
model.layers[-1].activation = None
print("[DEBUG] Model loaded successfully.")

last_conv_layer_name = ""
for layer in reversed(model.layers):
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
        last_conv_layer_name = layer.name
        break
print("[DEBUG] Automatically found last convolutional layer: '{}'".format(last_conv_layer_name))

print("[DEBUG] Loading and splitting data manifest...")
imagePaths = sorted(list(paths.list_images(DATASET_PATH)))
data = [(path, os.path.basename(os.path.dirname(path))) for path in imagePaths]
df = pd.DataFrame(data, columns=['filepath', 'class'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle is important for sampling
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
all_splits = list(skf.split(df['filepath'], df['class']))
_, val_index = all_splits[FOLD_TO_USE - 1]
val_df = df.iloc[val_index]

fire_df = val_df[val_df['class'] == '1'].sample(NUM_IMAGES // 2, random_state=42)
nofire_df = val_df[val_df['class'] == '0'].sample(NUM_IMAGES // 2, random_state=42)
sample_df = pd.concat([fire_df, nofire_df])

# ===================================================================
# === THE CRITICAL FIX IS APPLIED HERE ===
# We rename the 'class' column because 'class' is a reserved keyword in Python
# and cannot be used as an attribute (e.g., row.class).
sample_df = sample_df.rename(columns={'class': 'class_label'})
# ===================================================================

if sample_df.empty:
    raise ValueError("[ERROR] The sample DataFrame is empty. Could not find images to process.")
else:
    print("[DEBUG] Selected {} images for visualization.".format(len(sample_df)))

# ==============================================================================
# 4. GENERATE AND SAVE THE VISUALIZATION
# ==============================================================================
print("[INFO] Starting image processing loop to generate Grad-CAMs...")
fig, axes = plt.subplots(NUM_IMAGES, 2, figsize=(8, NUM_IMAGES * 4))
fig.suptitle('Grad-CAM Explainability Analysis', fontsize=20)

for i, row in enumerate(sample_df.itertuples()):
    print("[DEBUG] Processing image {} of {}: {}".format(i + 1, len(sample_df), row.filepath))
    img = load_img(row.filepath, target_size=(NORM_SIZE, NORM_SIZE))
    img_array = img_to_array(img)
    img_array_scaled = np.expand_dims(img_array / 255.0, axis=0)

    preds = model.predict(img_array_scaled)
    pred_class_index = np.argmax(preds[0])
    pred_class_name = "Fire" if pred_class_index == 1 else "NoFire"

    heatmap = get_gradcam_heatmap(model, img_array_scaled, last_conv_layer_name, pred_index=pred_class_index)
    
    if heatmap is None:
        print("[FATAL] Could not generate heatmap for {}. Skipping this image.".format(row.filepath))
        continue

    gradcam_img = superimpose_gradcam(img_array, heatmap)

    # Plot original image, now using the safe attribute name 'class_label'
    axes[i, 0].imshow(img)
    axes[i, 0].set_title("Original (True: {})".format(getLabel(int(row.class_label))))
    axes[i, 0].axis('off')

    # Plot Grad-CAM image
    axes[i, 1].imshow(gradcam_img)
    axes[i, 1].set_title("Grad-CAM (Predicted: {})".format(pred_class_name))
    axes[i, 1].axis('off')

print("[DEBUG] Image processing loop finished.")
plt.tight_layout(rect=[0, 0, 1, 0.98])
save_path = "{}/final_report_gradcam.png".format(PLOT_DIR)

print("[INFO] Saving final figure to {}...".format(save_path))
plt.savefig(save_path, dpi=300)
# plt.show() is commented out to ensure it saves even on systems without a GUI
print("[INFO] Figure saved successfully.")

print("\n[INFO] Grad-CAM script finished.")