import pandas as pd
import os
import re

def clean_text(text):
    """Linguistic preprocessing: lowercase and remove non-alphabetic noise"""
    if not isinstance(text, str): 
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    return text.strip()

def process_liar_dataset():
    print("Step 1: Processing LIAR dataset...")
    # This matches the path created by your download_datasets.py
    path = "../data/raw/liar/train.csv" 
    
    if os.path.exists(path):
        # Read the CSV (your download script saved it with index=False)
        # LIAR format: Col 0=ID, Col 1=Label, Col 2=Statement/Text
        df = pd.read_csv(path)
        
        # Select the label and the text statement
        # Note: If your CSV has no headers, use df.iloc[:, [1, 2]]
        try:
            # LIAR usually has labels in the 2nd column and text in the 3rd
            df = df.iloc[:, [1, 2]]
            df.columns = ['label', 'text']
            
            # Binary Classification Mapping:
            # Fake/Misleading (0): false, pants-fire, barely-true
            # Real/Authentic (1): true, mostly-true, half-true
            fake_labels = ['false', 'pants-fire', 'barely-true']
            df['label'] = df['label'].apply(lambda x: 0 if str(x) in fake_labels else 1)
            
            print(f"✓ Loaded {len(df)} rows from LIAR")
            return df
        except Exception as e:
            print(f"Error parsing columns: {e}")
            return pd.DataFrame()
    else:
        print(f"Critical Error: File not found at {path}")
        return pd.DataFrame()

if __name__ == "__main__":
    # 1. Process Data
    liar_df = process_liar_dataset()
    
    if not liar_df.empty:
        # 2. Clean Text
        print("Step 2: Cleaning text content...")
        liar_df['text'] = liar_df['text'].apply(clean_text)
        
        # 3. Create Processed Directory
        processed_dir = "../data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        
        # 4. Save Final CSV
        output_file = os.path.join(processed_dir, "cleaned_data.csv")
        liar_df.to_csv(output_file, index=False)
        
        print("-" * 30)
        print(f"SUCCESS!")
        print(f"Total Records: {len(liar_df)}")
        print(f"File Saved: {output_file}")
        print("-" * 30)
    else:
        print("Process failed. Please check if 'data/raw/liar/train.csv' exists.")