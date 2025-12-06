import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_roi')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'metadata_roi.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'roi_model.keras')
RESULTS_DIR = os.path.join(BASE_DIR, 'notebooks', 'evaluation_results_roi')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('ROI Classification Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, 'roi_confusion_matrix.png'))
    plt.close()

def plot_roc(y_true, y_pred_prob):
    # For binary classification, we only need the probability of the positive class (Malignant)
    # Malignant is class '1'
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROI Model)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'roi_roc_curve.png'))
    plt.close()

def main():
    print("--- Evaluating ROI Model ---")
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train first.")
        return

    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    df['target'] = df['target'].astype(str)
    
    # Reconstruct the split (Must match training split logic!)
    unique_ids = df['refnum'].unique()
    split_idx = int(len(unique_ids) * 0.8)
    test_ids = unique_ids[split_idx:]
    test_df = df[df['refnum'].isin(test_ids)]
    
    print(f"Evaluating on {len(test_df)} patches.")

    # 2. Generator
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=DATA_PATH,
        x_col='filename',
        y_col='target',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False 
    )

    # 3. Load Model and Predict
    model = load_model(MODEL_PATH)
    probs = model.predict(test_gen)
    preds = np.argmax(probs, axis=1)
    y_true = test_gen.classes
    
    # Map classes
    # Generator sorts alphanumerically: '0' (Benign), '1' (Malignant)
    class_names = ['Benign', 'Malignant'] 
    
    # 4. Reports
    report = classification_report(y_true, preds, target_names=class_names)
    print("\nClassification Report:\n")
    print(report)
    
    with open(os.path.join(RESULTS_DIR, 'roi_report.txt'), 'w') as f:
        f.write(report)
        
    plot_confusion_matrix(y_true, preds, class_names)
    plot_roc(y_true, probs)
    
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()