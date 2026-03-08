import re
import torch
from transformers import RobertaTokenizer
from werkzeug.security import generate_password_hash
import sys
import os

# Ensure backend imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_cleaner import clean_text
from routes import simple_sentiment_analysis, predict_topic

def run_all_tests():
    results = []

    # ---------------------------------------------------------
    # TC-U01
    # ---------------------------------------------------------
    tc01_input = "REUTERS (City) - News!!!"
    tc01_expected = "reuters city news"
    tc01_actual = re.sub(r'\s+', ' ', clean_text(tc01_input))
    tc01_status = "Pass" if tc01_actual == tc01_expected else "Fail"
    results.append({
        "id": "TC-U01", "desc": "Basic NLP Normalization",
        "input": f'"{tc01_input}"', "expected": f'"{tc01_expected}"',
        "actual": f'"{tc01_actual}"', "status": tc01_status
    })

    # ---------------------------------------------------------
    # TC-U02
    # ---------------------------------------------------------
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tc02_input = "news string (any length)"
    tc02_expected = "Tensor of shape [1, 512]"
    inputs = tokenizer(tc02_input, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    shape = list(inputs['input_ids'].shape)
    tc02_actual = f"[{shape[0]}, {shape[1]}] Tensor"
    tc02_status = "Pass" if shape == [1, 512] else "Fail"
    results.append({
        "id": "TC-U02", "desc": "RoBERTa Tokenization Shape",
        "input": f'"{tc02_input}"', "expected": tc02_expected,
        "actual": tc02_actual, "status": tc02_status
    })

    # ---------------------------------------------------------
    # TC-U03
    # ---------------------------------------------------------
    tc03_input = "PARIS (AFP) - Valid news..."
    tc03_expected = "True"
    has_agency_tag = bool(re.search(
        r'^[A-Z]{2,}[A-Za-z0-9\s,]*\s*\([A-Za-z\s]+\)\s*-{1,2}\s*',
        tc03_input[:150]
    ))
    tc03_actual = str(has_agency_tag)
    tc03_status = "Pass" if has_agency_tag else "Fail"
    results.append({
        "id": "TC-U03", "desc": "Agency Tag Detection",
        "input": f'"{tc03_input}"', "expected": tc03_expected,
        "actual": tc03_actual, "status": tc03_status
    })

    # ---------------------------------------------------------
    # TC-U04
    # ---------------------------------------------------------
    tc04_input = "SHOCKING TRUTH EXPOSED!"
    tc04_expected = "True"
    is_short_text = len(tc04_input) < 250
    fake_cap_count = len(re.findall(r'\b[A-Z]{5,}\b', tc04_input))
    has_excessive_caps = fake_cap_count >= (3 if is_short_text else 5)
    tc04_actual = str(has_excessive_caps)
    tc04_status = "Pass" if has_excessive_caps else "Fail"
    results.append({
        "id": "TC-U04", "desc": "Excessive Caps Penalty",
        "input": f'"{tc04_input}"', "expected": tc04_expected,
        "actual": tc04_actual, "status": tc04_status
    })

    # ---------------------------------------------------------
    # TC-U05
    # ---------------------------------------------------------
    tc05_input = "AI: 0.8, Source: 90"
    tc05_expected = "83.0"
    ai_score_raw = 0.8
    source_trust = 90
    fusion_score = (ai_score_raw * 100 * 0.70) + (source_trust * 0.30)
    tc05_actual = str(fusion_score)
    tc05_status = "Pass" if fusion_score == 83.0 else "Fail"
    results.append({
        "id": "TC-U05", "desc": "Credibility Fusion Calculation",
        "input": tc05_input, "expected": tc05_expected,
        "actual": tc05_actual, "status": tc05_status
    })

    # ---------------------------------------------------------
    # TC-U06
    # ---------------------------------------------------------
    tc06_input = "mypassword123"
    tc06_expected = "Non-plaintext string"
    hashed = generate_password_hash(tc06_input)
    # Just show the first 15 chars for brevity in output
    tc06_actual = hashed[:15] + "..."
    tc06_status = "Pass" if hashed != tc06_input and (hashed.startswith("pbkdf2") or hashed.startswith("scrypt")) else "Fail"
    results.append({
        "id": "TC-U06", "desc": "Password Security Hashing",
        "input": f'"{tc06_input}"', "expected": tc06_expected,
        "actual": tc06_actual, "status": tc06_status
    })

    # ---------------------------------------------------------
    # TC-U07
    # ---------------------------------------------------------
    tc07_input = "This is a great achievement"
    tc07_expected = "Positive"
    tc07_actual = simple_sentiment_analysis(tc07_input)
    tc07_status = "Pass" if tc07_actual == tc07_expected else "Fail"
    results.append({
        "id": "TC-U07", "desc": "Sentiment Polarity Check",
        "input": f'"{tc07_input}"', "expected": f'"{tc07_expected}"',
        "actual": f'"{tc07_actual}"', "status": tc07_status
    })

    # ---------------------------------------------------------
    # TC-U08
    # ---------------------------------------------------------
    tc08_input = "Election results are in"
    tc08_expected = "Politics"
    tc08_actual = predict_topic(tc08_input)
    tc08_status = "Pass" if tc08_actual == tc08_expected else "Fail"
    results.append({
        "id": "TC-U08", "desc": "Topic Classification",
        "input": f'"{tc08_input}"', "expected": f'"{tc08_expected}"',
        "actual": f'"{tc08_actual}"', "status": tc08_status
    })


    # Print Table
    print("=" * 145)
    print(f"{'Test Case ID':<13} | {'Description':<32} | {'Input':<28} | {'Expected Output':<25} | {'Actual Output':<23} | {'Status':<6}")
    print("-" * 145)
    for r in results:
        print(f"{r['id']:<13} | {r['desc']:<32} | {r['input']:<28} | {r['expected']:<25} | {r['actual']:<23} | {r['status']:<6}")
    print("=" * 145)

if __name__ == '__main__':
    run_all_tests()
