import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from imutils import paths
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# --- !!! IMPORTANT: SET THE FOLD YOU WANT TO EVALUATE !!! ---
FOLD_TO_EVALUATE = 5  # Change this to 1, 2, 3, 4, or 5

# --- Paths and Parameters (should match your training script) ---
MODEL_NAME = "ProposedNetwork_StratifiedKFold_Final"
DATASET_PATH = './Dataset/FireNet_Augmented/'
MODEL_DIR = "./Model"
PLOT_DIR = "./Plot"

N_SPLITS = 5
NORM_SIZE = 224
BATCH_SIZE = 32

# Construct the model path based on the fold number
model_path = f"{MODEL_DIR}/{MODEL_NAME}_fold_{FOLD_TO_EVALUATE}.hdf5"
os.makedirs(PLOT_DIR, exist_ok=True)

# ==============================================================================
# 2. LOAD DATA AND RE-CREATE THE EXACT K-FOLD SPLIT
# ==============================================================================
print(f"[INFO] Evaluating model for Fold {FOLD_TO_EVALUATE} from path: {model_path}")

# Load all image filepaths and labels into a DataFrame
print("[INFO] Loading image paths and creating DataFrame...")
imagePaths = sorted(list(paths.list_images(DATASET_PATH)))
data = [(path, os.path.basename(os.path.dirname(path))) for path in imagePaths]
df = pd.DataFrame(data, columns=['filepath', 'class']).sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize StratifiedKFold with the SAME parameters as in training to get the same splits
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Find the specific validation set indices for the fold we want to evaluate
val_indices = None
for i, (train_index, val_index) in enumerate(skf.split(df['filepath'], df['class'])):
    if i + 1 == FOLD_TO_EVALUATE:
        val_indices = val_index
        break

if val_indices is None:
    raise ValueError(f"Could not find indices for Fold {FOLD_TO_EVALUATE}. Check N_SPLITS.")

# Create the validation DataFrame for the specified fold
validation_df = df.iloc[val_indices]
print(f"[INFO] Successfully isolated the validation dataset for Fold {FOLD_TO_EVALUATE} ({len(validation_df)} images).")

# ==============================================================================
# 3. CREATE GENERATOR, LOAD MODEL, AND PREDICT
# ==============================================================================
# Create a data generator for ONLY the validation data of the chosen fold
# IMPORTANT: We only rescale the data. No augmentation for evaluation.
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='filepath',
    y_col='class',
    target_size=(NORM_SIZE, NORM_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # CRITICAL: Do not shuffle for evaluation
)

# Load the saved model for the specified fold
print("[INFO] Loading model...")
model = load_model(model_path)

# Predict on the correct validation data
print("[INFO] Generating predictions...")
y_prob = model.predict(validation_generator, verbose=1)
y_pred = y_prob.argmax(axis=-1)  # Predicted class indices
y_true = validation_generator.classes  # True class labels

# ==============================================================================
# 4. GENERATE AND PLOT THE CONFUSION MATRIX
# ==============================================================================
# Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
labels = ['NoFire', 'Fire']

def plot_confusion_matrix_with_counts(cm, classes,
                                      normalize=False,
                                      title='Normalized Confusion Matrix with Counts',
                                      cmap=plt.cm.Blues,
                                      figsize=(12, 12),
                                      font_size=16,
                                      value_font_size=14,
                                      save_path=None):
    """
    Function to plot a confusion matrix with normalized and absolute values.
    """
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap, aspect='auto')  # Align color bar
    cbar = fig.colorbar(im, ax=ax)  # Add color bar
    cbar.ax.tick_params(labelsize=font_size - 2)  # Adjust color bar font size
    ax.set_title(title, fontsize=font_size)

    # Set tick marks and labels
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=font_size - 2)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=font_size - 2)

    # Add text annotations (normalized and absolute counts)
    fmt = '.2f' if normalize else 'd'
    thresh = cm_normalized.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        normalized_value = f"{cm_normalized[i, j]:.2f}"
        absolute_value = f"{cm[i, j]} of {int(cm[i, :].sum())} images"
        text_color = "white" if cm_normalized[i, j] > thresh else "black"
        ax.text(j, i, f"{normalized_value}\n{absolute_value}",
                ha="center", va="center", color=text_color, fontsize=value_font_size)

    ax.set_ylabel('True label', fontsize=font_size)
    ax.set_xlabel('Predicted label', fontsize=font_size)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

# Call the function to plot
plot_confusion_matrix_with_counts(
    cm,
    classes=labels,
    normalize=True,
    title='Normalized Confusion Matrix with Counts',
    cmap=plt.cm.Blues,
    figsize=(12, 12),  # Adjust size as needed
    font_size=18,
    value_font_size=22,
    save_path='./Plot/confusion_matrix_custom.png'
)