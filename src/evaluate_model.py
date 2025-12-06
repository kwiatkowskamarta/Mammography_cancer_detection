import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'metadata_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mammography_resnet.keras')
RESULTS_DIR = os.path.join(BASE_DIR, 'notebooks', 'evaluation_results')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Create results folder if not exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    """
    Generates and saves the Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def plot_roc_curves(y_true, y_pred_probs, class_names):
    """
    Generates and saves ROC curves for each class.
    """
    # Binarize the labels for multi-class ROC
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    save_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def visualize_predictions(generator, model, class_names, num_samples=9):
    """
    Saves a grid of images with True vs Predicted labels.
    """
    # Get a batch of images
    x_batch, y_batch = next(generator)
    
    # Predict
    preds = model.predict(x_batch)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_batch, axis=1)
    
    plt.figure(figsize=(15, 15))
    for i in range(min(num_samples, len(x_batch))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_batch[i])
        
        true_label = class_names[true_classes[i]]
        pred_label = class_names[pred_classes[i]]
        
        color = 'green' if true_classes[i] == pred_classes[i] else 'red'
        
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
        
    save_path = os.path.join(RESULTS_DIR, 'sample_predictions.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def main():
    print("--- Loading Data & Model ---")
    
    # 1. Load Metadata
    df = pd.read_csv(CSV_PATH)
    test_df = df[df['dataset'] == 'test'].copy()
    test_df['target'] = test_df['target'].astype(str) # Keras requirement
    
    print(f"Evaluating on {len(test_df)} Test images.")

    # 2. Setup Generator (Important: shuffle=False to match labels!)
    datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=DATA_PATH,
        x_col="filename",
        y_col="target",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False 
    )

    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Run train_model.py first.")
        return
        
    model = load_model(MODEL_PATH)
    
    # 4. Generate Predictions
    print("Generating predictions...")
    # predict() returns probabilities for each class
    y_pred_probs = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Get True labels from the generator
    y_true = test_generator.classes
    
    # Get Class Names (0, 1, 2) -> Map them if needed
    # Based on our previous script: 0=N, 1=B, 2=M
    # However, flow_from_dataframe sorts classes alphanumerically.
    # Let's verify mapping from generator
    class_indices = test_generator.class_indices
    # Invert dictionary to get {0: '0', 1: '1', 2: '2'}
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Map to human readable names
    # Assuming target was saved as 0, 1, 2 in preprocessing
    # 0 -> Normal, 1 -> Benign, 2 -> Malignant
    human_labels = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
    class_names = [human_labels[int(idx_to_class[i])] for i in range(len(class_indices))]
    
    print(f"Class Mapping: {class_indices} -> {class_names}")

    # 5. Generate Reports
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)
    
    # Save Report to text file
    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 6. Generate Plots
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    plot_roc_curves(y_true, y_pred_probs, class_names)
    
    # Visual predictions need a fresh batch
    test_generator.reset() # Reset to start
    visualize_predictions(test_generator, model, class_names)

    print(f"\nEvaluation Complete! Check the '{RESULTS_DIR}' folder.")

if __name__ == "__main__":
    main()