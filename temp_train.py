import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(img_width, img_height):
    """
    Create a Convolutional Neural Network for fish freshness classification
    
    Args:
        img_width (int): Image width
        img_height (int): Image height
    
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = Sequential([
        tf.keras.layers.Input(shape=(img_width, img_height, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 classes
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Configuration
    img_width, img_height = 128, 128
    batch_size = 32
    
    # Base directory containing train, test, and valid folders
    # IMPORTANT: Replace this with your actual path
    BASE_DIR = r'D:\Fish Freshness Detection\dataset'  # Use raw string for Windows paths
    
    # Data generators using directory-based approach
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),  # Path to training directory
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'valid'),  # Path to validation directory
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Print class information
    print("Classes found:")
    print(train_generator.class_indices)
    
    # Create and train the model
    model = create_model(img_width, img_height)
    
    # Add early stopping and model checkpointing
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_fish_freshness_model.h5',
        monitor='val_accuracy', 
        save_best_only=True
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=20,  # Increased epochs with early stopping
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Save final model
    model.save('fish_freshness_model.h5')
    
    print("Model training complete.")
    
    # Optional: Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

if __name__ == '__main__':
    main()