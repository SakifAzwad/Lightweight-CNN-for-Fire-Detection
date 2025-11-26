from keras.models import load_model
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Path to your saved model
model_path = "./Model/optimized_model_LAST.hdf5"  # Replace with your model's name
model = load_model(model_path)

# Dataset Parameters
dataset_path = "./Dataset/FireNet_Augmented/"  # Replace with your dataset path
norm_size = 224
batch_size = 32

# Data Generator for Validation
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.3)
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(norm_size, norm_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False  # Ensure labels and predictions align
)

y_true = validation_generator.classes  # True labels from the generator
y_pred_prob = model.predict(validation_generator, verbose=1)  # Predicted probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Predicted class indices

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Detailed classification report
report = classification_report(
    y_true, y_pred, target_names=['NoFire', 'Fire'], output_dict=True, zero_division=1
)
metrics_data = {
    'Metric': ['F1 Score', 'Precision', 'Recall'],
    'Fire': [
        round(report['Fire']['f1-score'], 3),
        round(report['Fire']['precision'], 3),
        round(report['Fire']['recall'], 3)
    ],
    'No Fire': [
        round(report['NoFire']['f1-score'], 3),
        round(report['NoFire']['precision'], 3),
        round(report['NoFire']['recall'], 3)
    ]
}

metrics_df = pd.DataFrame(metrics_data)

# Prepare data for the table
cell_text = [["Labels", "F1 Score", "Precision", "Recall"],  # Column Headers
             ["Fire", metrics_df['Fire'][0], metrics_df['Fire'][1], metrics_df['Fire'][2]],  # Fire Data Row
             ["No Fire", metrics_df['No Fire'][0], metrics_df['No Fire'][1], metrics_df['No Fire'][2]]]  # No Fire Data Row

# Set up figure and axis
fig, ax = plt.subplots(figsize=(8, 3))  # Adjust size as needed
ax.axis('off')

# Add the title text just above the table with minimal spacing
ax.text(0.50, 0.75, 'Classification Report', ha='center', va='top', fontsize=14, weight='bold', transform=ax.transAxes)

# Create the table
table = ax.table(
    cellText=cell_text,
    cellLoc='center',
    colWidths=[0.3, 0.2, 0.2, 0.2],
    loc='center'
)

# Adjust the main table styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)  # Scale up for readability

# Set bold for header row in the main table
for (i, j), cell in table.get_celld().items():
    if i == 0:  # Header Row Styling
        cell.set_text_props(weight='bold')

plt.show()

name_file = "optimized_model"
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Define the getLabel function
def getLabel(id):
    return ['NoFire', 'Fire'][id]

# Helper function to get distinct balanced samples from the validation generator
def get_balanced_samples(generator, num_samples_per_class=6):
    """
    Collect balanced samples from the generator.
    Args:
        generator: Validation generator.
        num_samples_per_class: Number of samples to select per class.
    Returns:
        images: Array of balanced images.
        labels: Array of corresponding labels.
    """
    images, labels = [], []
    class_counts = {0: 0, 1: 0}  # Track counts for 'NoFire' and 'Fire'
    while class_counts[0] < num_samples_per_class or class_counts[1] < num_samples_per_class:
        batch_data, batch_labels = next(generator)  # Get a batch
        for i in range(len(batch_data)):
            label = np.argmax(batch_labels[i])  # Convert one-hot encoding to class index
            if class_counts[label] < num_samples_per_class:
                images.append(batch_data[i])
                labels.append(label)
                class_counts[label] += 1
                if class_counts[0] == num_samples_per_class and class_counts[1] == num_samples_per_class:
                    break
    return np.array(images), np.array(labels)

# Load the best model
# model_path = "./Model/" + name_file + ".hdf5"
# model = load_model(model_path, compile=False)

# Get a balanced batch of validation data
X_test, y_test = get_balanced_samples(validation_generator, num_samples_per_class=6)

# Make predictions
res = model.predict(X_test).argmax(axis=-1)

# Visualize predictions for the best model
plt.figure(figsize=(12, 12))
for i in range(12):  # Display all 12 images
    plt.subplot(3, 4, i + 1)  # Create a 3x4 grid of plots

    # Convert data to a valid range for imshow
    image = (X_test[i] * 255).astype(np.uint8)  # Denormalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    plt.imshow(image)
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel('Prediction = %s' % getLabel(res[i]), fontsize=10)

# Save and display the plot
plt.tight_layout()  # Adjust spacing for the 3x4 grid
plt.savefig('./Plot/' + name_file + '_Best.png', dpi=300)
plt.show()