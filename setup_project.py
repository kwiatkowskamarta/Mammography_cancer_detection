import os

def create_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created: {path}")

def setup_structure():
    # define Directory Structure
    dirs = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'models'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Directory created: {d}")

    # create requirements.txt
    requirements = """pandas
numpy
opencv-python
matplotlib
scikit-learn
tqdm
jupyter
"""
    create_file('requirements.txt', requirements)

    # create .gitignore
    gitignore = """
# Python
__pycache__/
*.py[cod]
venv/
.env

# jupyter
.ipynb_checkpoints

# data
data/
models/
*.pgm
*.png
"""
    create_file('.gitignore', gitignore)

    # create README.md
    readme = """# Mammography Tumor Detection (MIAS Dataset)

## Project Overview
This project aims to detect tumors in mammography scans using Machine Learning.
It utilizes the MIAS dataset, performing preprocessing, augmentation, and classification.

## Structure
- `data/`: Contains raw and processed image data.
- `src/`: Source code for preprocessing and modeling.
- `notebooks/`: Jupyter notebooks for analysis.

## Usage
1. Place MIAS `.pgm` images and `info.txt` in `data/raw/`.
2. Run `python src/data_preprocessing.py` to clean and augment data.
"""
    create_file('README.md', readme)

    # create the main preprocessing script (src/data_preprocessing.py)
    preprocessing_code = r'''import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_DATA_PATH = 'data/raw'
PROCESSED_PATH = 'data/processed'
IMG_SIZE = (224, 224)
INFO_FILE = 'info.txt'
FINAL_CSV_NAME = 'metadata_processed.csv'

def parse_info_file(file_path):
    """
    parses the MIAS info.txt file which has variable column lengths.
    structure: REFNUM BG CLASS SEVERITY X Y RADIUS
    note: 'NORM' lines lack Severity, X, Y, Radius.
    """
    print(f"Reading metadata from: {file_path}")
    
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue # Skip empty lines
        
        refnum = parts[0]
        bg_tissue = parts[1]
        cls = parts[2] # 'CIRC', 'SPIC', 'MISC', 'ARCH', 'ASYM', 'NORM'
        
        if cls == 'NORM':
            severity = 'N' # Normal
            x, y, radius = 0, 0, 0
        else:
            # check if line is malformed or missing data
            if len(parts) < 7:
                continue 
            severity = parts[3]
            x = int(parts[4])
            y = int(parts[5])
            radius = int(parts[6])
            
        data.append([refnum, bg_tissue, cls, severity, x, y, radius])
        
    df = pd.DataFrame(data, columns=['refnum', 'bg_tissue', 'class', 'severity', 'x', 'y', 'radius'])
    return df

def clean_metadata(df):
    """
    cleans metadata: handles duplicates by keeping the most severe case per image.
    severity Map: M (Malignant) > B (Benign) > N (Normal)
    """
    # map severity to a numeric score for sorting
    severity_map = {'M': 2, 'B': 1, 'N': 0}
    df['severity_score'] = df['severity'].map(severity_map)
    
    # sort descending by severity and remove duplicates based on 'refnum'
    #this keeps the most severe diagnosis if an image appears multiple times
    df_clean = df.sort_values('severity_score', ascending=False).drop_duplicates('refnum')
    
    df_clean = df_clean.drop(columns=['severity_score'])
    print(f"Metadata cleaned. Unique images: {len(df_clean)}")
    return df_clean

def process_and_augment(df):
    """
    reads images, resizes them, and performs augmentation (flips).
    saves processed images as PNG.
    """
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)
        
    new_metadata = []
    
    print("Starting image processing and augmentation...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_id = row['refnum']
        img_path = os.path.join(RAW_DATA_PATH, f"{img_id}.pgm")
        
        if not os.path.exists(img_path):
            #try searching for file with spaces or different extension (just in case)
            continue
            
        #read Image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # resize
        img_resized = cv2.resize(img, IMG_SIZE)
        
        # --- AUGMENTATION STRATEGY ---
        # 1. original
        # 2. horizontal Flip
        # 3. vertical Flip
        
        transforms = {
            'orig': None,
            'hflip': 1,
            'vflip': 0
        }
        
        for suffix, code in transforms.items():
            if suffix == 'orig':
                final_img = img_resized
            else:
                final_img = cv2.flip(img_resized, code)
                
            # save
            new_filename = f"{img_id}_{suffix}.png"
            save_path = os.path.join(PROCESSED_PATH, new_filename)
            cv2.imwrite(save_path, final_img)
            
            # update metadata
            new_row = row.copy()
            new_row['filename'] = new_filename
            new_row['augmentation'] = suffix
            new_metadata.append(new_row)
            
    #save final CSV
    final_df = pd.DataFrame(new_metadata)
    final_csv_path = os.path.join('data', FINAL_CSV_NAME)
    final_df.to_csv(final_csv_path, index=False)
    print(f"Processing complete. Data saved to {final_csv_path}")
    print(f"Total images generated: {len(final_df)}")

if __name__ == "__main__":
    info_path = os.path.join(RAW_DATA_PATH, INFO_FILE)
    
    if os.path.exists(info_path):
        df_raw = parse_info_file(info_path)
        df_clean = clean_metadata(df_raw)
        process_and_augment(df_clean)
    else:
        print(f"Error: {INFO_FILE} not found in {RAW_DATA_PATH}")
'''
    create_file('src/data_preprocessing.py', preprocessing_code)

    print("\n--- Project Setup Complete ---")
    print("Please run: pip install -r requirements.txt")
    print("Run the processor: python src/data_preprocessing.py")

if __name__ == "__main__":
    setup_structure()
