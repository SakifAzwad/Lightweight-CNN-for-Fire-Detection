# ==============================================================================
# 1. IMPORTS AND SETUP
# ==============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import cv2
import tensorflow as tf
from imutils import paths
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Import your model architecture
from cnn_multi import combine_4SC_2_1_V3

# Suppress warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel('ERROR')

# ==============================================================================
# 2. CONFIGURATION PARAMETERS
# ==============================================================================
DATASET_PATH = './Dataset/ThermalFire/'
MODEL_NAME = "ProposedNetwork_StratifiedKFold_Final"
NORM_SIZE = 224
NUM_CLASSES = 2
INPUT_SHAPE = (NORM_SIZE, NORM_SIZE, 3)
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
N_SPLITS = 5  # Number of folds for cross-validation

# --- Define Output Directories ---
LOGS_DIR = "./Logs"
MODEL_DIR = "./Model"
PLOT_DIR = "./Plot"

# --- Create Output Directories if they don't exist ---
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def calculate_gflops(model, batch_size=1):
    """Calculates GFLOPS for a Keras model."""
    input_shape = (batch_size,) + model.input_shape[1:]
    inputs = tf.random.normal(input_shape)
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(inputs)
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops / 1e9

def get_model_parameters(model):
    """Counts the total parameters of a model."""
    return model.count_params()

def getLabel(class_id):
    """Returns the string label for a given class ID."""
    return ['NoFire', 'Fire'][class_id]

def plot_confusion_matrix_with_counts(cm, classes, title='Normalized Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
    """Plots a detailed confusion matrix with both normalized values and absolute counts."""
    cm_counts = cm.astype(int)
    cm_sum = cm_counts.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm_counts.astype('float'), cm_sum, out=np.zeros_like(cm_counts, dtype=float), where=(cm_sum!=0))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}\n({cm_counts[i, j]} images)",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# ==============================================================================
# 4. DATA PREPARATION
# ==============================================================================
print("[INFO] Loading image paths and labels...")
imagePaths = sorted(list(paths.list_images(DATASET_PATH)))
data = [(path, os.path.basename(os.path.dirname(path))) for path in imagePaths]
df = pd.DataFrame(data, columns=['filepath', 'class']).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"[INFO] Found {len(df)} images. Class distribution:\n{df['class'].value_counts()}")

# ==============================================================================
# 5. ONE-TIME MODEL ANALYSIS (GFLOPS & PARAMETERS)
# ==============================================================================
print("\n[INFO] Performing one-time model analysis...")
temp_model = combine_4SC_2_1_V3(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
total_params = get_model_parameters(temp_model)
gflops = calculate_gflops(temp_model, batch_size=BATCH_SIZE)
print(f"[INFO] Model Parameters: {total_params:,}")
print(f"[INFO] GFLOPS: {gflops:.4f}")
del temp_model # Clean up memory

# ==============================================================================
# 6. STRATIFIED K-FOLD CROSS-VALIDATION
# ==============================================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
all_scores = { 'accuracy': [], 'loss': [], 'precision_macro': [], 'recall_macro': [], 'f1_macro': [], 'mcc': [] }
fold_num = 1
history_last_fold = None
best_model_path_last_fold = ""
val_index_last_fold = None

for train_index, val_index in skf.split(df['filepath'], df['class']):
    print(f"\n========== FOLD {fold_num}/{N_SPLITS} ==========")
    train_df, val_df = df.iloc[train_index], df.iloc[val_index]

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filepath', y_col='class', target_size=(NORM_SIZE, NORM_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
    validation_generator = val_datagen.flow_from_dataframe(val_df, x_col='filepath', y_col='class', target_size=(NORM_SIZE, NORM_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

    model = combine_4SC_2_1_V3(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

    # These paths are now unique for each fold
    model_filepath = f"{MODEL_DIR}/{MODEL_NAME}_fold_{fold_num}.hdf5"
    csv_log_path = f"{LOGS_DIR}/{MODEL_NAME}_fold_{fold_num}_training.csv"
    
    callbacks_list = [
        ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
        CSVLogger(csv_log_path) # The CSVLogger will save the log for this fold here
    ]

    print(f"[INFO] Training model for Fold {fold_num}...")
    history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, callbacks=callbacks_list, verbose=1)

    print(f"[INFO] Evaluating Fold {fold_num}...")
    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    y_true, y_pred = validation_generator.classes, np.argmax(model.predict(validation_generator), axis=1)
    
    report = classification_report(y_true, y_pred, target_names=['NoFire', 'Fire'], output_dict=True)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    all_scores['loss'].append(loss); all_scores['accuracy'].append(accuracy)
    all_scores['precision_macro'].append(report['macro avg']['precision'])
    all_scores['recall_macro'].append(report['macro avg']['recall'])
    all_scores['f1_macro'].append(report['macro avg']['f1-score']); all_scores['mcc'].append(mcc)

    print(f"[RESULT] Fold {fold_num} -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}")
    
    if fold_num == N_SPLITS: # Save data from the last fold for final visualizations
        history_last_fold = history
        best_model_path_last_fold = model_filepath
        val_index_last_fold = val_index

    fold_num += 1

# ==============================================================================
# 7. AGGREGATE AND DISPLAY FINAL RESULTS
# ==============================================================================
print("\n========== CROSS-VALIDATION FINAL RESULTS ==========")
print(f"Metrics reported as: Mean ± Standard Deviation over {N_SPLITS} folds.\n")
for metric, scores in all_scores.items():
    print(f"{metric.replace('_', ' ').title():<20}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print("======================================================")

# ==============================================================================
# 8. VISUALIZATION (using the last trained fold as a representative example)
# ==============================================================================
print("\n[INFO] Generating visualizations for the last trained fold...")
if history_last_fold:
    # --- Plot Accuracy and Loss Curves ---
    plt.style.use("seaborn-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history_last_fold.history['accuracy'], label='Train Acc'); ax1.plot(history_last_fold.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend(loc='lower right')
    ax2.plot(history_last_fold.history['loss'], label='Train Loss'); ax2.plot(history_last_fold.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/{MODEL_NAME}_curves.png", dpi=300); plt.show()

    # --- Plot Confusion Matrix ---
    model.load_weights(best_model_path_last_fold)
    val_df_last_fold = df.iloc[val_index_last_fold]
    validation_generator_last_fold = val_datagen.flow_from_dataframe(val_df_last_fold, x_col='filepath', y_col='class', target_size=(NORM_SIZE, NORM_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
    y_true = validation_generator_last_fold.classes
    y_pred = np.argmax(model.predict(validation_generator_last_fold), axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_with_counts(cm, classes=['NoFire', 'Fire'], save_path=f"{PLOT_DIR}/{MODEL_NAME}_cm.png")

    # --- Plot Classification Report Table ---
    report = classification_report(y_true, y_pred, target_names=['NoFire', 'Fire'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=report_df.round(3).values, colLabels=report_df.columns, rowLabels=report_df.index, cellLoc = 'center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.2)
    ax.set_title("Classification Report (Last Fold)", pad=20, fontsize=16)
    plt.savefig(f"{PLOT_DIR}/{MODEL_NAME}_report.png", dpi=300, bbox_inches='tight'); plt.show()

    # --- Plot Prediction Examples ---
    images, _ = next(validation_generator_last_fold)
    predictions = np.argmax(model.predict(images), axis=1)
    plt.figure(figsize=(15, 15))
    for i in range(min(12, len(images))):
        plt.subplot(4, 3, i + 1)
        rgb_image = cv2.cvtColor((images[i]*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.title(f"Prediction: {getLabel(predictions[i])}")
        plt.axis('off')
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/{MODEL_NAME}_predictions.png", dpi=300); plt.show()

print("[INFO] Script finished successfully.")