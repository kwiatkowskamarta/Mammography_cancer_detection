import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
ROI_PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed_roi')
INFO_FILE = 'info.txt'
ROI_SIZE = (224, 224) 

def parse_info_file(file_path):
    #same parser as before
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue
        refnum, bg, cls = parts[0], parts[1], parts[2]
        if cls == 'NORM': continue # Skip Normals for ROI experiment
        if len(parts) < 7: continue
        severity = parts[3]
        x, y, radius = int(parts[4]), int(parts[5]), int(parts[6])
        data.append([refnum, cls, severity, x, y, radius])
    return pd.DataFrame(data, columns=['refnum', 'class', 'severity', 'x', 'y', 'radius'])

def extract_and_save_rois(df):
    if not os.path.exists(ROI_PROCESSED_PATH):
        os.makedirs(ROI_PROCESSED_PATH)
    
    new_metadata = []
    
    print(f"Extracting ROIs for {len(df)} abnormalities...")
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_id = row['refnum']
        img_path = os.path.join(RAW_DATA_PATH, f"{img_id}.pgm")
        
        if not os.path.exists(img_path): continue
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        
        # coordinates from info.txt
        # MIAS documentation: "The x-axis increases from left to right, the y-axis increases from bottom to top" - flipping the Y coordinate: y_cv2 = height - y_mias
        center_x = row['x']
        center_y = h - row['y'] 
        radius = row['radius']
        
        # croping a square box around the circle (1.2x radius)
        box_size = int(radius * 1.2)
        
        y1 = max(0, center_y - box_size)
        y2 = min(h, center_y + box_size)
        x1 = max(0, center_x - box_size)
        x2 = min(w, center_x + box_size)
        
        roi = img[y1:y2, x1:x2]
        
        #resize to 224x224 for the model
        if roi.size == 0: continue
        roi_resized = cv2.resize(roi, ROI_SIZE)
        
        # enhance contrast (CLAHE) (important for textures)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        roi_enhanced = clahe.apply(roi_resized)
        
        # augmemtation
        transforms = {
            'orig': None,
            'hflip': 1,
            'vflip': 0,
            'rot90': 90
        }
        
        for suffix, code in transforms.items():
            if code == 90:
                final_img = cv2.rotate(roi_enhanced, cv2.ROTATE_90_CLOCKWISE)
            elif code is not None:
                final_img = cv2.flip(roi_enhanced, code)
            else:
                final_img = roi_enhanced
            
            # save
            filename = f"{img_id}_{suffix}.png"
            cv2.imwrite(os.path.join(ROI_PROCESSED_PATH, filename), final_img)
            
            meta_row = row.copy()
            meta_row['filename'] = filename
            new_metadata.append(meta_row)

    # save CSV
    final_df = pd.DataFrame(new_metadata)
    # map Severity to binary target: B=0, M=1
    final_df['target'] = final_df['severity'].map({'B': 0, 'M': 1})
    
    csv_path = os.path.join(BASE_DIR, 'data', 'metadata_roi.csv')
    final_df.to_csv(csv_path, index=False)
    print(f"ROI Dataset created: {len(final_df)} patches saved to {csv_path}")

if __name__ == "__main__":
    info_path = os.path.join(RAW_DATA_PATH, INFO_FILE)
    df = parse_info_file(info_path)
    extract_and_save_rois(df)
