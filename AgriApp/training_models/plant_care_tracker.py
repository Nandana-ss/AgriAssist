from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os

# Define the paths to your datasets
dataset_paths = ['AgriApp/data/progress/rice_dataset']

# Define image size and batch size
img_size = (150, 150) 
batch_size = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Create a list of training and validation generators for each dataset
train_generators = []
validation_generators = []

# Initialize a variable to collect class indices
class_indices = None

for dataset_path in dataset_paths:
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )
    train_generators.append(train_generator)
    validation_generators.append(validation_generator)
    
    if class_indices is None:
        class_indices = train_generator.class_indices
    else:
        # Check if class indices match across datasets
        if class_indices != train_generator.class_indices:
            raise ValueError("Class indices do not match across datasets")

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_indices), activation='softmax')  # Using consistent number of classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model
epochs = 1
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for train_generator, validation_generator in zip(train_generators, validation_generators):
        model.fit(
            train_generator,
            epochs=1,  # Train for one epoch at a time
            validation_data=validation_generator,
            verbose=1
        )

# Define the path to save the model
model_save_path = 'AgriApp/models/plant_growth_model.keras'

# Create the 'models' directory if it doesn't exist
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the model
model.save(model_save_path)