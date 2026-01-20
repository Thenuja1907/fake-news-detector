import pandas as pd
import os
import re

def clean_text(text):
    """NLP Preprocessing: Lowercase and remove noise"""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove numbers and special chars
    return text.strip()

def process_liar_dataset():
    print("Checking for raw data...")
    # This assumes your data is in fake-news-detector/data/raw/liar/
    path = "../data/raw/liar/train.csv" 
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Select Label (col 1) and Statement (col 2)
        df = df.iloc[:, [1, 2]]
        df.columns = ['label', 'text']
        
        # Map 6 labels to 0 (Fake) and 1 (Real)
        fake_labels = ['false', 'pants-fire', 'barely-true']
        df['label'] = df['label'].apply(lambda x: 0 if str(x).strip() in fake_labels else 1)
        
        print(f"✓ Successfully loaded {len(df)} records.")
        return df
    else:
        print(f"Error: File not found at {os.path.abspath(path)}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = process_liar_dataset()
    if not df.empty:
        print("Cleaning text data...")
        df['text'] = df['text'].apply(clean_text)
        
        # Save to processed folder
        output_dir = "../data/processed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "cleaned_data.csv")
        
        df.to_csv(output_path, index=False)
        print(f"✓ SUCCESS! Cleaned data saved to: {output_path}")