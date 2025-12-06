import pandas as pd
import numpy as np
import os
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
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_roi')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'metadata_roi.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'roi_model.keras')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16 
EPOCHS = 30
LR = 0.0001

def add_sample_weights(df):
    """
    Calculates class weights and adds a 'sample_weight' column to the DataFrame.
    """
    # Get class indices (0 or 1)
    y = df['target'].astype(int).values
    
    # Compute weights (Balanced)
    # Formula: n_samples / (n_classes * n_samples_j)
    weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y), 
        y=y
    )
    weights_dict = dict(enumerate(weights))
    print(f"Computed Class Weights (0=Benign, 1=Malignant): {weights_dict}")
    
    # Map weights to a new column
    df['sample_weight'] = df['target'].astype(int).map(weights_dict)
    return df

def main():
    if not os.path.exists(CSV_PATH):
        print("Error: metadata_roi.csv not found. Run create_roi_dataset.py first.")
        return

    # 1. Load and Prepare Data
    df = pd.read_csv(CSV_PATH)
    df['target'] = df['target'].astype(str) # Keras requires string labels
    
    # Add Sample Weights to the DataFrame
    df = add_sample_weights(df)
    
    # We simply split by first 80% for train, last 20% for test (simplification for ROI)
    # Since we augmented, we want to keep patient groups together ideally, 
    # but for this specific patch-based experiment, a simple split is often acceptable 
    # if we augmented *after* splitting. 
    # However, since we augmented *before*, let's rely on the fact that augmentation 
    # created consecutive files.
    
    # Group by Refnum to avoid leakage
    unique_ids = df['refnum'].unique()
    split_idx = int(len(unique_ids) * 0.8)
    train_ids = unique_ids[:split_idx]
    test_ids = unique_ids[split_idx:]
    
    train_df = df[df['refnum'].isin(train_ids)]
    test_df = df[df['refnum'].isin(test_ids)]
    
    print(f"Training on {len(train_df)} patches. Testing on {len(test_df)} patches.")

    # 2. Generators
    # Note: We use 'weight_col' for the Training generator
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=DATA_PATH,
        x_col='filename',
        y_col='target',
        weight_col='sample_weight', # <--- FIX IS HERE
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=DATA_PATH,
        x_col='filename',
        y_col='target',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # 3. Model Architecture
    base = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False 
    
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x) # 2 classes: Benign, Malignant
    
    model = Model(base.input, output)
    
    model.compile(optimizer=Adam(LR), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', 'auc']) # 'auc' is safe in Keras 3
    
    # 4. Train
    print("--- Starting Training ---")
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=EPOCHS,
        # class_weight is REMOVED (handled by weight_col above)
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')
        ]
    )
    
    print("ROI Model Trained Successfully.")

if __name__ == "__main__":
    main()