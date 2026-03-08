import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

model_path = "models/roberta_fake_news"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

tests = [
    ("REAL", "WASHINGTON (Reuters) - The Senate voted on Tuesday to pass a major infrastructure spending bill with bipartisan support from both parties."),
    ("REAL", "LONDON (Reuters) - Britain raised interest rates for the fourth time in a row on Thursday as its central bank tries to combat the fastest inflation in 30 years."),
    ("FAKE", "SHOCKING: Government scientists EXPOSED secretly adding mind-control chemicals to the water supply! SHARE before they DELETE this!"),
    ("FAKE", "BREAKING!! Deep State plans to STEAL the election! Mainstream media won't tell you the TRUTH! Wake up America!!"),
]

print(f"{'Expected':<8} {'Predicted':<12} {'Real%':<10} {'Fake%'}")
print("-" * 50)
for label, text in tests:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]
    real_p = probs[0].item()
    fake_p = probs[1].item()
    predicted = "REAL" if real_p > fake_p else "FAKE"
    match = "✅" if predicted == label else "❌"
    print(f"{label:<8} {predicted:<12} {real_p:.4f}     {fake_p:.4f}  {match}")
