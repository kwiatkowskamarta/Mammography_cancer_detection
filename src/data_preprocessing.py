import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
RAW_DATA_PATH = 'data/raw'
PROCESSED_PATH = 'data/processed'
IMG_SIZE = (224, 224)
INFO_FILE = 'info.txt'
FINAL_CSV_NAME = 'metadata_processed.csv'
RANDOM_SEED = 42

def parse_info_file(file_path):
    """
    parses the MIAS info.txt file.
    handles 'NORM' cases by filling missing values with defaults.
    """
    print(f"Reading metadata from: {file_path}")
    data = []
    
    #read file line by line to handle variable length
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue 
            
        refnum = parts[0]
        bg_tissue = parts[1]
        cls = parts[2] # CIRC, SPIC, MISC, ARCH, ASYM, NORM
        
        if cls == 'NORM':
            severity = 'N' 
            x, y, radius = 0, 0, 0
        else:
            if len(parts) < 7: continue
            severity = parts[3]
            x, y, radius = int(parts[4]), int(parts[5]), int(parts[6])
            
        data.append([refnum, bg_tissue, cls, severity, x, y, radius])
        
    df = pd.DataFrame(data, columns=['refnum', 'bg_tissue', 'class', 'severity', 'x', 'y', 'radius'])
    return df

def clean_and_split_data(df):
    """
    removes duplicates (keeps most severe diagnosis).
    encodes target labels (0: Normal, 1: Benign, 2: Malignant).
    splits data into Train (80%) and Test (20%) ensuring NO PATIENT OVERLAP.
    """
    #handle duplicates: malignant > benign > normal
    severity_map = {'M': 2, 'B': 1, 'N': 0}
    df['severity_score'] = df['severity'].map(severity_map)
    df_clean = df.sort_values('severity_score', ascending=False).drop_duplicates('refnum').copy()
    
    # add target column for training
    df_clean['target'] = df_clean['severity_score']
    
    # patient-level split
    #splitting based on unique REFNUMs to prevent data leakage
    unique_patients = df_clean['refnum'].values
    train_ids, test_ids = train_test_split(
        unique_patients, 
        test_size=0.20, 
        random_state=RANDOM_SEED,
        stratify=df_clean['severity'] #ensures equal % of cancer/normal in both sets
    )
    
    # asign split label
    df_clean['dataset'] = 'train'
    df_clean.loc[df_clean['refnum'].isin(test_ids), 'dataset'] = 'test'
    
    print(f"Data Split: {len(train_ids)} Train patients, {len(test_ids)} Test patients.")
    return df_clean

def apply_clahe(img):
    """
    applies Contrast Limited Adaptive Histogram Equalization.
    enhances local contrast to make tissue structures more visible.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def process_and_augment(df):
    """
    pipeline: load -> resize -> CLAHE ->augment -> save
    """
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)
        
    final_metadata = []
    
    print("Starting processing pipeline (Resize + CLAHE + Augmentation)...")
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_id = row['refnum']
        img_path = os.path.join(RAW_DATA_PATH, f"{img_id}.pgm")
        
        if not os.path.exists(img_path):
            continue
            
        # load Image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 1. resize (to reduce computation)
        img_resized = cv2.resize(img, IMG_SIZE)
        
        # 2. apply CLAHE (Contrast Enhancement)
        # Applying after resize is faster and usually sufficient
        img_enhanced = apply_clahe(img_resized)
        
        # 3. augmentation Strategy
        # only augmenting the TRAINING set to keep the test set "pure" (like real world data).
       
        if row['dataset'] == 'train':
            transforms = {'orig': None, 'hflip': 1, 'vflip': 0}
        else:
            transforms = {'orig': None} # No augmentation for test set
            
        for suffix, code in transforms.items():
            if suffix == 'orig':
                final_img = img_enhanced
            else:
                final_img = cv2.flip(img_enhanced, code)
            
            # save file
            new_filename = f"{img_id}_{suffix}.png"
            save_path = os.path.join(PROCESSED_PATH, new_filename)
            cv2.imwrite(save_path, final_img)
            
            # record metadata
            new_row = row.copy()
            new_row['filename'] = new_filename
            new_row['augmentation'] = suffix
            final_metadata.append(new_row)

    #save CSV
    final_df = pd.DataFrame(final_metadata)
    final_csv_path = os.path.join('data', FINAL_CSV_NAME)
    final_df.to_csv(final_csv_path, index=False)
    
    print(f"done! Processed data saved to {final_csv_path}")
    print(f"Total images: {len(final_df)}")
    print(f"Class distribution:\n{final_df['severity'].value_counts()}")

if __name__ == "__main__":
    info_path = os.path.join(RAW_DATA_PATH, INFO_FILE)
    
    if os.path.exists(info_path):
        df_raw = parse_info_file(info_path)
        df_split = clean_and_split_data(df_raw)
        process_and_augment(df_split)
    else:
        print(f"Error: {INFO_FILE} not found in {RAW_DATA_PATH}. Please verify setup.")
