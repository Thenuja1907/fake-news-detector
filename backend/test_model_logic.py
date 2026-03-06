import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os

def test_model():
    model_path = "models/roberta_fake_news"
    if not os.path.exists(model_path):
        print("Model path not found.")
        return

    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval()

    test_sentences = [
        "The sun is a star in the center of our solar system.", # Reality/Real
        "Secret lizard people are controlling the global banking system from Mars." # Fake
    ]

    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        
        fake_prob = probs[0][1].item()
        real_prob = probs[0][0].item()
        
        print(f"Text: {text[:50]}...")
        print(f"Index 0 (Real?): {real_prob:.4f}")
        print(f"Index 1 (Fake?): {fake_prob:.4f}")
        print(f"Classification logic says is_fake = {fake_prob > real_prob}")
        print("-" * 20)

if __name__ == "__main__":
    test_model()
