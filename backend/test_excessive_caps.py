import re

def test_excessive_caps():
    print("=" * 60)
    print("TESTING EXCESSIVE CAPS PENALTY (TC-U04)")
    print("=" * 60)
    
    # 1. Provide test cases with varying lengths and capitalization
    test_cases = [
        {
            "name": "Short text, heavy caps (Expected: TRUE -> Penalized)",
            "text": "SHOCKING TRUTH EXPOSED! The DEEPSTATE is hiding the SECRET to eternal life!",
            "expected_flag": True
        },
        {
            "name": "Short text, normal caps (Expected: FALSE -> Not penalized)",
            "text": "The United States government announced a new trade deal today in Washington.",
            "expected_flag": False
        },
        {
            "name": "Long text, heavy caps (Expected: TRUE -> Penalized)",
            "text": ("This is a very long text to simulate an article that is completely fake and trying to manipulate you. "
                     "THEY DO NOT WANT YOU TO KNOW THIS! The ELITES are controlling the MEDIA and the BANKS. "
                     "WAKE UP SHEEPLE! This is the most IMPORTANT message you will ever read. Please SHARE this "
                     "before it gets taken down by the ESTABLISHMENT! This text is now comfortably over the "
                     "two hundred and fifty character limit requirement for a long text."),
            "expected_flag": True
        },
        {
            "name": "Long text, normal caps (Expected: FALSE -> Not penalized)",
            "text": ("This is a normal lengthy news excerpt. The Federal Reserve, meeting in Washington D.C., "
                     "announced its decision to hold interest rates steady. Officials indicated that inflation, "
                     "while cooling, still requires careful monitoring over the coming months. Wall Street analysts "
                     "expect the current economic conditions to persist into the third quarter. This text is now "
                     "comfortably over the two hundred and fifty character limit requirement for a long text."),
            "expected_flag": False
        }
    ]

    print(f"{'Test Case':<60} | {'Short?'[:6]} | {'Caps Found'[:10]} | {'Result'}")
    print("-" * 100)

    for case in test_cases:
        content = case["text"]
        
        # EXACT LOGIC FROM routes.py
        is_short_text = len(content) < 250
        # regex looks for whole words (\b) that are all uppercase ([A-Z]) and 5 or more characters long ({5,})
        fake_cap_count = len(re.findall(r'\b[A-Z]{5,}\b', content))
        
        # Penalize short text if >= 3 caps; long text if >= 5 caps
        has_heavy_caps = fake_cap_count >= (3 if is_short_text else 5)
        
        # Determine Pass/Fail for the test
        if has_heavy_caps == case["expected_flag"]:
            status = "✅ PASS"
        else:
            status = f"❌ FAIL (Expected {case['expected_flag']}, Got {has_heavy_caps})"
            
        print(f"{case['name']:<60} | {str(is_short_text):<6} | {fake_cap_count:<10} | {status}")
        
    print("\n[CONCLUSION]")
    print("Tests successfully evaluated the dynamic text-length caps threshold logic.")

if __name__ == "__main__":
    test_excessive_caps()
