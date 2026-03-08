import requests
import time

def test_api():
    base_url = "http://127.0.0.1:5000/analyze"
    tests = [
        ("WASHINGTON (Reuters) - The U.S. House of Representatives voted on Tuesday to pass a tax reform bill.", "Real News"),
        ("BREAKING: They don't want you to know this but the deep state has been hiding the real cure for cancer. Share before they delete this!", "Fake News"),
        ("LONDON (Reuters) - Britain's government confirmed on Monday it will seek to renegotiate its trade deal with the European Union, according to officials.", "Real News"),
        ("SHOCKING: The mainstream media won't tell you about the miracle cure that big pharma is hiding from the public. Wake up sheeple!", "Fake News"),
        ("21st Century Wire - The globalist agenda exposed. What they are hiding will DESTROY your understanding of reality. 100% proof inside.", "Fake News")
    ]

    for text, expected in tests:
        response = requests.post(base_url, json={"content": text})
        res = response.json()
        print(f"Text: {text[:60]}...")
        if res.get("status") == "success":
            print(f"Classification: {res.get('classification')} | Score: {res.get('credibility_score')} | Expected: {expected}")
            data = res.get('data', {})
            metadata = data.get('metadata', {})
            print(f"Reasoning: {metadata.get('reasoning')}")
        else:
            print(f"Error: {res}")
        print("-" * 50)

if __name__ == "__main__":
    test_api()
