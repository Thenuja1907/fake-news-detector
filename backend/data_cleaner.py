# backend/src/preprocessing/data_cleaner.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords"""
        return [word for word in tokens if word not in self.stop_words]
    
    def preprocess_dataset(self, input_path, output_path):
        """Preprocess entire dataset"""
        df = pd.read_csv(input_path)
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Handle missing values
        df = df.dropna(subset=['cleaned_text', 'label'])
        
        # Save processed data
        df.to_csv(output_path, index=False)
        print(f"✓ Preprocessed data saved to {output_path}")
        
        return df

# Usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_dataset(
        '../data/raw/liar/train.csv',
        '../data/processed/train_cleaned.csv'
    )