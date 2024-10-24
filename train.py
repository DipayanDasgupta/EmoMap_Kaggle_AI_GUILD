import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# Load the CSV file
data = pd.read_csv('train_dataset.csv')

# Extract pixel values and labels
pixels = data['pixels'].tolist()
labels = data['emotion'].values

# Convert the pixel strings to 48x48 numpy arrays and normalize pixel values
def process_images(pixels):
    images = np.array([np.fromstring(pixel, sep=' ').reshape(48, 48) for pixel in pixels])
    images = images / 255.0  # Normalize the pixel values to be between 0 and 1
    images = images.reshape(images.shape[0], 48, 48, 1)  # Add channel dimension
    return images

images = process_images(pixels)

# Example visualization
plt.imshow(images[0].reshape(48, 48), cmap='gray')
plt.show()
# Set up the data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator to the image data
datagen.fit(images)
# Build the CNN model
def create_model():
    model = Sequential()

    # Add convolutional, max pooling, and dropout layers
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))

    return model

model = create_model()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
# Split your data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Set batch size and epochs
batch_size = 64
epochs = 50

# Set up the data generator with augmentation
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

# Train the model
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    steps_per_epoch=X_train.shape[0] // batch_size)

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Load and process test.csv
test_data = pd.read_csv('test.csv')
test_pixels = test_data['pixels'].tolist()
test_images = process_images(test_pixels)

# Generate predictions for the test data
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Prepare the submission file
submission = pd.DataFrame({'id': test_data['id'], 'emotion': predicted_classes})
submission.to_csv('submission.csv', index=False)
