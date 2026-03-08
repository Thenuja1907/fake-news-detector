import requests

base_url = 'http://127.0.0.1:5000/analyze'

texts = [
    "Federal Reserve officials said in a statement that interest rates would remain unchanged at their next meeting, citing stable inflation.",
    "The prime minister announced the new budget today. Officials said the new policy will be implemented next week according to the government."
]

for text in texts:
    response = requests.post(base_url, json={'content': text})
    res = response.json()
    if res.get('status') == 'success':
        print(f"Classification: {res.get('classification')} | Score: {res.get('credibility_score')}")
    else:
        print("Failed")
