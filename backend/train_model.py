import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_system():
    print("Loading cleaned data...")
    # Matches the path from your successful data_cleaner.py run
    data_path = "../data/processed/cleaned_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Cleaned data not found at {data_path}!")
        return

    # Load data
    df = pd.read_csv(data_path).dropna()
    
    # NLP Vectorization (TF-IDF)
    # This converts words into numbers the AI can understand
    print("Extracting features (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['text'])
    y = df['label']

    # Split Data: 80% for training, 20% for testing accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    print("Training AI Model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Create 'models' folder and save the brain files
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fake_news_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    
    # Calculate Accuracy
    accuracy = model.score(X_test, y_test)
    print("-" * 30)
    print(f"✓ SUCCESS: Model Trained!")
    print(f"✓ Accuracy: {round(accuracy*100, 2)}%")
    print(f"✓ Saved: models/fake_news_model.pkl")
    print("-" * 30)

if __name__ == "__main__":
    train_system()