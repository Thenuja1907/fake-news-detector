import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from routes import simple_sentiment_analysis, predict_topic

def test_sentiment_and_topic():
    print("=" * 70)
    print("TESTING SENTIMENT (TC-U07) & TOPIC CLASSIFICATION (TC-U08)")
    print("=" * 70)
    
    # ---------------------------------------------------------
    # TC-U07: Sentiment Polarity Check
    # ---------------------------------------------------------
    print("\n--- 🧠 TC-U07: Sentiment Polarity Check ---")
    
    # These map to the positive/negative word arrays in simple_sentiment_analysis
    sentiment_cases = [
        {"desc": "Positive Language", "text": "This is a great achievement and excellent news!", "expected": "Positive"},
        {"desc": "Negative Language", "text": "A terrible disaster struck, causing tragic destruction.", "expected": "Negative"},
        {"desc": "Neutral Language",  "text": "The committee met on Tuesday to review standard procedures.", "expected": "Neutral"},
        {"desc": "Mixed Language",    "text": "It was a wonderful day until the terrible accident happened.", "expected": "Neutral"}, # 1 pos, 1 neg = 0
    ]
    
    for case in sentiment_cases:
        actual = simple_sentiment_analysis(case["text"])
        status = "✅ PASS" if actual == case["expected"] else f"❌ FAIL"
        print(f"{status} | Expected: {case['expected']:<8} | Got: {actual:<8} | Case: {case['desc']}")


    # ---------------------------------------------------------
    # TC-U08: Topic Classification
    # ---------------------------------------------------------
    print("\n--- 📚 TC-U08: Topic Classification ---")
    
    # These map to the topic keyword dictionaries in predict_topic
    topic_cases = [
        {"desc": "Politics Keywords",   "text": "Election results are in, the president addressed parliament.", "expected": "Politics"},
        {"desc": "Economy Keywords",    "text": "The stock market crashed after the bank hiked interest rates.", "expected": "Economy"},
        {"desc": "Health Keywords",     "text": "A new viral outbreak threatens public health, vaccine needed.", "expected": "Health"},
        {"desc": "Technology Keywords", "text": "The new software update features AI and revolutionary algorithms.", "expected": "Technology"},
        {"desc": "No Keywords Match",   "text": "My dog ate my homework yesterday afternoon in the park.", "expected": "General"},
    ]
    
    for case in topic_cases:
        actual = predict_topic(case["text"])
        status = "✅ PASS" if actual == case["expected"] else f"❌ FAIL"
        print(f"{status} | Expected: {case['expected']:<10} | Got: {actual:<10} | Case: {case['desc']}")

    print("\n[CONCLUSION]")
    print("Tests successfully evaluated the keyword extraction logic mapped in routes.py")

if __name__ == "__main__":
    test_sentiment_and_topic()
