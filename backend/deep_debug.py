"""
Deep diagnostic: test classification with Kaggle-style real and fake news examples.
Run from backend/ directory.
"""
import joblib
import torch
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

# ---- Kaggle True.csv style (Reuters datelines) ----
KAGGLE_REAL = [
    "WASHINGTON (Reuters) - The U.S. House of Representatives voted on Tuesday to pass a tax reform bill, with Republicans saying the legislation would boost the economy.",
    "LONDON (Reuters) - Britain's government confirmed on Monday it will seek to renegotiate its trade deal with the European Union, according to officials.",
    "NEW YORK (AP) - Federal Reserve officials said in a statement that interest rates would remain unchanged at their next meeting, citing stable inflation.",
]

# ---- Kaggle Fake.csv style (sensationalist / conspiracy) ----
KAGGLE_FAKE = [
    "BREAKING: They don't want you to know this but the deep state has been hiding the real cure for cancer. Share before they delete this!",
    "SHOCKING: The mainstream media won't tell you about the miracle cure that big pharma is hiding from the public. Wake up sheeple!",
    "21st Century Wire - The globalist agenda exposed. What they are hiding will DESTROY your understanding of reality. 100% proof inside.",
]

sk_model_path = r"models/fake_news_model.pkl"
tfidf_path    = r"models/tfidf_vectorizer.pkl"
roberta_path  = r"models/roberta_fake_news"

print("=" * 60)
print("SKLEARN MODEL")
print("=" * 60)
sk_model = joblib.load(sk_model_path)
tfidf    = joblib.load(tfidf_path)
classes  = list(sk_model.classes_)
print(f"classes_: {classes}")
print(f"Meaning: class {classes[0]} = idx-0  |  class {classes[1]} = idx-1")

for label, samples in [("REAL", KAGGLE_REAL), ("FAKE", KAGGLE_FAKE)]:
    print(f"\n--- Expected: {label} ---")
    for text in samples:
        vec   = tfidf.transform([text])
        proba = sk_model.predict_proba(vec)[0]
        pred  = sk_model.predict(vec)[0]
        print(f"  Text: {text[:60]}...")
        print(f"  proba={[round(p,3) for p in proba]}  predict={pred}")

print()
print("=" * 60)
print("ROBERTA MODEL  (config: 0=REAL, 1=FAKE)")
print("=" * 60)
tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
model     = RobertaForSequenceClassification.from_pretrained(roberta_path)
model.eval()

for label, samples in [("REAL", KAGGLE_REAL), ("FAKE", KAGGLE_FAKE)]:
    print(f"\n--- Expected: {label} ---")
    for text in samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = F.softmax(logits, dim=1)[0].tolist()
        r_real = probs[0]   # id2label[0] = REAL
        r_fake = probs[1]   # id2label[1] = FAKE
        pred   = "REAL" if r_real > r_fake else "FAKE"
        print(f"  Text: {text[:60]}...")
        print(f"  P(REAL)={r_real:.3f}  P(FAKE)={r_fake:.3f}  → {pred}")
