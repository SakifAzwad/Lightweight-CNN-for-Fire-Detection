from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import tensorflow as tf

# Paths
dataset_path = './Dataset/FireNet/'  # Original dataset path
augmented_dataset_path = './Dataset/FireNet_Augmented/'  # Path to save augmented images

# Ensure augmented directory exists
os.makedirs(augmented_dataset_path, exist_ok=True)

# Augmentation parameters
augmentation_multiplier = 3  # Number of augmented images per original image
target_size = (224, 224)  # Resize dimensions

# Define augmentation function
def augment_image(image):
    """Apply GPU-accelerated augmentations."""
    image = tf.image.random_flip_left_right(image)  # Horizontal flip
    image = tf.image.random_flip_up_down(image)    # Vertical flip
    image = tf.image.random_brightness(image, max_delta=0.3)  # Random brightness
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)  # Random contrast
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)  # Random saturation
    image = tf.image.random_hue(image, max_delta=0.1)  # Random hue
    image = tf.image.resize_with_crop_or_pad(image, target_size[0] + 10, target_size[1] + 10)  # Add padding
    image = tf.image.random_crop(image, size=[target_size[0], target_size[1], 3])  # Random crop
    image = tf.clip_by_value(image, 0.0, 1.0)  # Ensure pixel values are between [0, 1]
    return image

# Process and save augmented images
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if not os.path.isdir(class_path):
        continue
    
    # Create corresponding folder for augmented images
    augmented_class_path = os.path.join(augmented_dataset_path, class_folder)
    os.makedirs(augmented_class_path, exist_ok=True)
    
    for img_idx, img_name in enumerate(os.listdir(class_path), start=1):  # Track image number with img_idx
        img_path = os.path.join(class_path, img_name)
        
        try:
            # Load and preprocess the image
            raw_image = tf.io.read_file(img_path)
            image = tf.image.decode_image(raw_image, channels=3)
            image = tf.image.resize(image, target_size) / 255.0  # Normalize to [0, 1]
            
            # Print progress
            print(f"Processing Image {img_idx}: {img_name}")
            
            # Save the original image (optional, if you want it in the augmented dataset)
            tf.keras.preprocessing.image.save_img(
                os.path.join(augmented_class_path, img_name), image.numpy()
            )
            
            # Generate augmented images
            for i in range(augmentation_multiplier):
                augmented_image = augment_image(image)
                tf.keras.preprocessing.image.save_img(
                    os.path.join(augmented_class_path, f"aug_{i}_{img_name}"), augmented_image.numpy()
                )
                print(f"  Augmented Image {i+1} saved for {img_name}")
        
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

print("GPU-accelerated augmentation complete!")
