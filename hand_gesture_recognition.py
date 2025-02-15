# Suppress TensorFlow logs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs

# Import necessary libraries
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Define paths and categories
data_dir = '.'  # Current directory (leapGestRecog folder)
categories = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

# Step 2: Load images and labels
images = []
labels = []

for category in categories:
    # Loop through all session folders (01/, 02/, etc.)
    for session in os.listdir(data_dir):
        session_path = os.path.join(data_dir, session)
        if os.path.isdir(session_path):  # Check if it's a directory
            category_path = os.path.join(session_path, category)
            if os.path.exists(category_path):  # Check if the category folder exists
                class_num = categories.index(category)
                for img in os.listdir(category_path):
                    img_path = os.path.join(category_path, img)
                    try:
                        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                        if img_array is not None:
                            img_array = cv2.resize(img_array, (100, 100))  # Resize image to 100x100
                            images.append(img_array)
                            labels.append(class_num)
                        else:
                            print(f"Warning: Image {img_path} could not be read.")
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")

# Check if any images were loaded
if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images were loaded. Please check the directory structure and image files.")

# Convert to numpy arrays
images = np.array(images).reshape(-1, 100, 100, 1)  # Reshape for CNN input (100x100x1)
labels = np.array(labels)

# Normalize the images (scale pixel values to [0, 1])
images = images / 255.0

# Step 3: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(categories))
y_val = to_categorical(y_val, num_classes=len(categories))

# Step 4: Build the CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(len(categories), activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Step 6: Evaluate the model
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')

# Step 7: Save the model
model.save('hand_gesture_model.h5')
print("Model saved as 'hand_gesture_model.h5'")

# Step 8: Plot training history (optional)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()