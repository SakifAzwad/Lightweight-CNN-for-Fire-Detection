import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2
import tensorflow as tf
from imutils import paths
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================================================================
# 1. CONFIGURATION - !!! ONLY EDIT THIS SECTION !!!
# ==============================================================================
# --- Choose which fold's visualizations to generate as a representative example ---
FOLD_FOR_VISUALIZATION = 5  # <--- You can change this to 1, 2, 3, 4, or 5

# --- Paths and Parameters (MUST MATCH YOUR TRAINING SCRIPT) ---
MODEL_NAME = "ProposedNetwork_StratifiedKFold_Final"
DATASET_PATH = './Dataset/FireNet_Augmented/'
MODEL_DIR = "./Model"
PLOT_DIR = "./Plot"

N_SPLITS = 5
NORM_SIZE = 224
BATCH_SIZE = 32

# ==============================================================================
# DO NOT EDIT BELOW THIS LINE
# ==============================================================================

# --- Setup and Verification ---
os.makedirs(PLOT_DIR, exist_ok=True)
print("[INFO] Starting the comprehensive report generation process...")

# ==============================================================================
# 2. HELPER FUNCTIONS FOR VISUALIZATION
# ==============================================================================
def getLabel(class_id):
    """Returns the string label for a given class ID."""
    return ['NoFire', 'Fire'][class_id]

def plot_confusion_matrix_with_counts(cm, classes, title, save_path):
    """Plots a detailed confusion matrix with both normalized values and absolute counts."""
    cm_counts = cm.astype(int)
    cm_sum = cm_counts.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm_counts.astype('float'), cm_sum, out=np.zeros_like(cm_counts, dtype=float), where=(cm_sum!=0))
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "{:.2f}\n({} images)".format(cm_normalized[i, j], cm_counts[i, j]),
                    ha="center", va="center", color="white" if cm_normalized[i, j] > thresh else "black", fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# ==============================================================================
# 3. LOAD DATA AND RE-CREATE K-FOLD SPLITS
# ==============================================================================
print("[INFO] Loading image paths and recreating K-Fold splits...")
imagePaths = sorted(list(paths.list_images(DATASET_PATH)))
data = [(path, os.path.basename(os.path.dirname(path))) for path in imagePaths]
df = pd.DataFrame(data, columns=['filepath', 'class']).sample(frac=1, random_state=42).reset_index(drop=True)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
all_splits = list(skf.split(df['filepath'], df['class']))

# ==============================================================================
# 4. EVALUATE ALL FOLDS TO GET AGGREGATED METRICS
# ==============================================================================
all_scores = { 'accuracy': [], 'precision_macro': [], 'recall_macro': [], 'f1_macro': [], 'mcc': [] }

for i in range(N_SPLITS):
    fold_num = i + 1
    print("\n--- Evaluating Fold {}/{} ---".format(fold_num, N_SPLITS))
    
    # Isolate the validation data for this fold
    _, val_index = all_splits[i]
    val_df = df.iloc[val_index]

    # Create generator for this fold's validation set
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df, x_col='filepath', y_col='class',
        target_size=(NORM_SIZE, NORM_SIZE), batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False
    )

    # Load the corresponding model
    model_path = "{}/{}_fold_{}.hdf5".format(MODEL_DIR, MODEL_NAME, fold_num)
    if not os.path.exists(model_path):
        print("WARNING: Model file not found for Fold {} at {}. Skipping.".format(fold_num, model_path))
        continue
    model = load_model(model_path)
    
    # Predict and calculate metrics
    y_true = validation_generator.classes
    y_pred = np.argmax(model.predict(validation_generator, verbose=0), axis=1)
    
    # Use macro averaging and calculate MCC
    report = classification_report(y_true, y_pred, output_dict=True)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Store the required metrics
    all_scores['accuracy'].append(report['accuracy'])
    all_scores['precision_macro'].append(report['macro avg']['precision'])
    all_scores['recall_macro'].append(report['macro avg']['recall'])
    all_scores['f1_macro'].append(report['macro avg']['f1-score'])
    all_scores['mcc'].append(mcc)
    
    print("Fold {} Accuracy: {:.4f}, MCC: {:.4f}".format(fold_num, report['accuracy'], mcc))

# ==============================================================================
# 5. DISPLAY THE FINAL AGGREGATED REPORT (FOR YOUR PAPER'S TABLE)
# ==============================================================================
print("\n========== FINAL CROSS-VALIDATION REPORT (Mean ± Std. Dev.) ==========")
for metric, scores in all_scores.items():
    if scores: # Check if there are any scores to average
        mean = np.mean(scores)
        std = np.std(scores)
        # Using ljust for clean alignment
        print("{:<25}: {:.4f} ± {:.4f}".format(metric.replace('_', ' ').title(), mean, std))
print("======================================================================")

# ==============================================================================
# 6. GENERATE VISUALIZATIONS FOR THE REPRESENTATIVE FOLD (FOR YOUR PAPER'S FIGURES)
# ==============================================================================
print("\n[INFO] Generating visualizations for representative Fold {}...".format(FOLD_FOR_VISUALIZATION))

# Isolate data for the chosen visualization fold
_, val_index_vis = all_splits[FOLD_FOR_VISUALIZATION - 1]
val_df_vis = df.iloc[val_index_vis]
validation_generator_vis = val_datagen.flow_from_dataframe(
    dataframe=val_df_vis, x_col='filepath', y_col='class',
    target_size=(NORM_SIZE, NORM_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

# Load the model and get predictions
model_path_vis = "{}/{}_fold_{}.hdf5".format(MODEL_DIR, MODEL_NAME, FOLD_FOR_VISUALIZATION)
model_vis = load_model(model_path_vis)
y_true_vis = validation_generator_vis.classes
y_pred_vis = np.argmax(model_vis.predict(validation_generator_vis, verbose=0), axis=1)

# --- Plot 1: Detailed Confusion Matrix ---
cm = confusion_matrix(y_true_vis, y_pred_vis)
plot_confusion_matrix_with_counts(cm, classes=['NoFire', 'Fire'],
                                  title='Confusion Matrix (Fold {})'.format(FOLD_FOR_VISUALIZATION),
                                  save_path='{}/final_report_cm.png'.format(PLOT_DIR))

# --- Plot 2: Classification Report as a Table (from your code) ---
report_vis = classification_report(y_true_vis, y_pred_vis, target_names=['NoFire', 'Fire'], output_dict=True)
report_df = pd.DataFrame(report_vis).transpose()
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight'); ax.axis('off')
table = ax.table(cellText=report_df.round(3).values, colLabels=report_df.columns, rowLabels=report_df.index, cellLoc='center', loc='center')
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.2)
ax.set_title("Classification Report (Fold {})".format(FOLD_FOR_VISUALIZATION), pad=20, fontsize=16)
plt.savefig("{}/final_report_table.png".format(PLOT_DIR), dpi=300, bbox_inches='tight'); plt.show()

# --- Plot 3: Prediction Examples (from your code) ---
images, _ = next(validation_generator_vis)
predictions = np.argmax(model_vis.predict(images), axis=1)
plt.figure(figsize=(15, 15))
for i in range(min(12, len(images))):
    plt.subplot(4, 3, i + 1)
    rgb_image = cv2.cvtColor((images[i]*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.title("Prediction: {}".format(getLabel(predictions[i])))
    plt.axis('off')
plt.tight_layout(); plt.savefig("{}/final_report_predictions.png".format(PLOT_DIR), dpi=300); plt.show()

print("\n[INFO] All reports and visualizations have been generated successfully.")