import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'metadata_processed.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'mammography_resnet.keras')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

def load_data():
    df = pd.read_csv(CSV_PATH)
    df['target'] = df['target'].astype(str)
    return df

def apply_class_weights_to_dataframe(df):
    """
    Calculates class weights and assigns a specific 'sample_weight' 
    to every row in the dataframe.
    """
    # 1. Calculate weights for the classes (0, 1, 2)
    y_train = df['target'].astype(int).values
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    weights_dict = dict(enumerate(class_weights))
    print(f"Computed Class Weights: {weights_dict}")

    # 2. Map these weights to a new column in the dataframe
    # Assuming target is string '0', '1', '2', we map to int first
    df['sample_weight'] = df['target'].astype(int).map(weights_dict)
    
    return df

def build_model(num_classes):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'auc']) # 'auc' is string literal in recent Keras
    
    return model

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plot_path = os.path.join(BASE_DIR, 'notebooks', 'training_history.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

def main():
    # 1. Load Data
    df = load_data()
    
    # 2. Split into Train/Test subsets based on the flag we created earlier
    train_df = df[df['dataset'] == 'train'].copy()
    test_df = df[df['dataset'] == 'test'].copy()
    
    # 3. Apply Weights directly to the DataFrame (The Fix)
    train_df = apply_class_weights_to_dataframe(train_df)
    
    print(f"Training on {len(train_df)} images, Validation on {len(test_df)} images.")

    # 4. Generators
    datagen = ImageDataGenerator(rescale=1./255)

    # Note: We add 'weight_col' here. The generator will now yield (x, y, sample_weight)
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=DATA_PATH,
        x_col="filename",
        y_col="target",
        weight_col="sample_weight",  # <--- CRITICAL CHANGE
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    valid_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=DATA_PATH,
        x_col="filename",
        y_col="target",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # 5. Build and Train
    model = build_model(num_classes=3)
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')
    ]

    print("--- Starting Training ---")
    # We REMOVED class_weight argument here because it's handled by the generator now
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plot_history(history)
    print("Training finished successfully.")

if __name__ == "__main__":
    main()