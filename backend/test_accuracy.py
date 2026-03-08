import requests
import json

def test_accurate_classification():
    base_url = "http://127.0.0.1:5000/analyze"
    
    test_cases = [
        {
            "name": "Case 1: Short Real News (Ambiguous to Rules but Formal)",
            "content": "The Federal Reserve chairman announced that interest rates will remain unchanged to support ongoing economic growth. This decision was based on current inflation data.",
            "expected": "Real News"
        },
        {
            "name": "Case 2: Fake News with Formal Words (Tries to trick rules)",
            "content": "Official government reports confirmed that a secret cabal is manipulating the harvest. Researchers found unbelievable proof of deep state intervention. Share this truth now!",
            "expected": "Fake News"
        },
        {
            "name": "Case 3: Standard Real News (With Agency Tag)",
            "content": "WASHINGTON (Reuters) - The U.S. State Department announced new diplomatic efforts in the Middle East to stabilize oil regions.",
            "expected": "Real News"
        },
        {
            "name": "Case 4: Obvious Fake News (Sensationalist)",
            "content": "SHOCKING!!! THEY ARE HIDING THE MIRACLE CURE FOR EVERYTHING!!! BIG PHARMA DOESN'T WANT YOU TO WATCH THIS VIDEO!!!",
            "expected": "Fake News"
        }
    ]

    print("=" * 100)
    print(f"{'Test Case':<60} | {'Expected':<12} | {'Actual':<12} | {'Score'}")
    print("-" * 100)

    for case in test_cases:
        try:
            response = requests.post(base_url, json={"content": case["content"]})
            data = response.json()
            actual = data.get("classification", "Error")
            score = data.get("credibility_score", 0.0)
            
            status = "✅ PASS" if actual == case["expected"] else "❌ FAIL"
            print(f"{case['name']:<60} | {case['expected']:<12} | {actual:<12} | {score}% {status}")
        except Exception as e:
            print(f"{case['name']:<60} | Error connecting: {e}")

    print("=" * 100)

if __name__ == "__main__":
    test_accurate_classification()
