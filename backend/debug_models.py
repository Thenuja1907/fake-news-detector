import joblib
import torch
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

def test_models():
    sk_model_path = r"c:\Users\imple\fake-news-detector\backend\models\fake_news_model.pkl"
    tfidf_path = r"c:\Users\imple\fake-news-detector\backend\models\tfidf_vectorizer.pkl"
    roberta_path = r"c:\Users\imple\fake-news-detector\backend\models\roberta_fake_news"
    
    print("--- Sklearn (TF-IDF) Model Info ---")
    if os.path.exists(sk_model_path):
        sk_model = joblib.load(sk_model_path)
        print(f"Classes: {sk_model.classes_}")
        tfidf = joblib.load(tfidf_path)
        
        texts = [
            "This is a verified news report with facts.",
            "SHOCKING BREAKTHROUGH: Miracle cure found for everything!"
        ]
        
        for t in texts:
            vec = tfidf.transform([t])
            probs = sk_model.predict_proba(vec)[0]
            print(f"Text: {t[:30]}...")
            print(f"Probs: {probs}")
            
    print("\n--- RoBERTa Model Info ---")
    if os.path.exists(roberta_path):
        tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        model = RobertaForSequenceClassification.from_pretrained(roberta_path)
        model.eval()
        
        texts = [
            "The economy grew by 2% today.",
            "Aliens have taken over the White House!"
        ]
        
        for t in texts:
            inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]
            print(f"Text: {t[:30]}...")
            print(f"Probs: {probs.tolist()}")

if __name__ == "__main__":
    test_models()
