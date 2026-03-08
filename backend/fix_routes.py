import re

with open('routes.py', 'r', encoding='utf-8') as f:
    src = f.read()

# -----------------------------------------------------------------------
# Locate the forensic rule engine block by unique start/end markers
# -----------------------------------------------------------------------
start_marker = "        # 2. FORENSIC RULE ENGINE"
end_marker   = "        # 3a. Sklearn Baseline Model"

idx_start = src.find(start_marker)
idx_end   = src.find(end_marker)

if idx_start == -1 or idx_end == -1:
    print("ERROR: Could not locate forensic block")
    print("  start found:", idx_start != -1)
    print("  end found:  ", idx_end   != -1)
    exit(1)

print(f"Block found: chars {idx_start}–{idx_end}")

# -----------------------------------------------------------------------
# New forensic block  (LF line endings to match the file)
# -----------------------------------------------------------------------
new_block = """\
        # 2. FORENSIC RULE ENGINE (Primary — Rule-First Architecture)
        # The RoBERTa model outputs ~50/50 (under-trained), so rules drive the decision.
        is_short_text = len(content) < 250

        # --- Real News Signals ---
        # Dateline pattern: CITY (Agency) - ... (strongest real indicator)
        has_agency_marker = bool(re.search(
            r'^[A-Z]{2,}[A-Za-z0-9\\s,]*\\s*\\([A-Za-z\\s]+\\)\\s*-{1,2}\\s*',
            content[:150]
        ))

        # STRONG real signals — multi-word phrases from formal journalism (+0.07 each)
        real_keywords_strong = [
            "according to", "officials said", "said in a statement",
            "said on monday", "said on tuesday", "said on wednesday",
            "said on thursday", "said on friday", "in a statement",
            "press conference", "press release", "spokesperson said",
            "study found", "researchers found", "confirmed by",
            "told reporters", "told journalists", "the associated press",
            "reuters reported", "bbc reported", "per reuters", "per the ap"
        ]
        # WEAK real signals — individual credibility words (+0.03 each)
        real_keywords_weak = [
            "confirmed", "announced", "reported", "stated", "government",
            "official", "minister", "parliament", "court", "police said",
            "authorities", "agency", "spokesperson", "published", "journal"
        ]
        real_hits_strong = sum(1 for k in real_keywords_strong if k in content.lower())
        real_hits_weak   = sum(1 for k in real_keywords_weak   if k in content.lower())

        # --- Fake News Signals ---
        # Excessive capitalisation (5+ letter all-caps words)
        fake_cap_count = len(re.findall(r'\\b[A-Z]{5,}\\b', content))
        has_heavy_caps = fake_cap_count > (8 if is_short_text else 3)

        # Unambiguous sensational / manipulation phrases (strong fake signal)
        sensational_phrases = [
            "they don't want you to know", "share before they delete",
            "mainstream media won't tell you", "wake up sheeple",
            "secret revealed", "banned truth", "miracle cure",
            "big pharma", "they are hiding", "fake media",
            "you won't see this on cnn", "warning!!!", "must watch",
            "go viral", "100% proof", "deep state conspiracy",
            "the real truth", "what they're hiding"
        ]
        # Milder clickbait words — weaker signals on their own
        mild_sensational = [
            "shocking", "exposed", "scandal", "unbelievable",
            "won't believe", "truth about", "deep state",
            "mainstream media", "they don't want", "the truth is",
            "click here", "share this"
        ]
        sensational_hit      = sum(1 for p in sensational_phrases if p in content.lower())
        mild_sensational_hit = sum(1 for p in mild_sensational   if p in content.lower())
        # Strong fake: 1+ unambiguous phrase  OR  3+ mild words together
        is_sensationalist = (sensational_hit >= 1) or (mild_sensational_hit >= 3)

        # Excessive punctuation (!!! ???)
        excl_count = content.count('!') + content.count('?')
        has_excessive_punct = excl_count > 5

        # --- Build Score from Rules (0.0 – 1.0) ---
        # Start at 0.48 — just below the 0.52 Real threshold.
        # One weak real-signal word tips neutral text to Real News.
        # Strong fake signals push comfortably below the threshold.
        real_prob = 0.48

        if has_agency_marker:
            real_prob = 0.94  # Hard override — verified agency dateline
        else:
            # Add for real signals
            real_prob += min(real_hits_strong * 0.07, 0.21)   # up to +0.21
            real_prob += min(real_hits_weak   * 0.03, 0.12)   # up to +0.12
            # Deduct for fake signals
            if has_heavy_caps:               real_prob -= 0.20
            if is_sensationalist:            real_prob -= 0.25
            elif mild_sensational_hit >= 1:  real_prob -= 0.10
            if has_excessive_punct:          real_prob -= 0.10

"""

# Patch
new_src = src[:idx_start] + new_block + src[idx_end:]

# -----------------------------------------------------------------------
# Also fix the is_sensationalist references in the red_flags section
# and the classification threshold (0.55 -> 0.52)
# -----------------------------------------------------------------------
# Update red_flags block: replace the old reference to sensational_hit
new_src = new_src.replace(
    'if is_sensationalist and "sensationalism" not in red_flags:\n            red_flags.append("sensationalism")',
    'if (is_sensationalist or mild_sensational_hit >= 1) and "sensationalism" not in red_flags:\n            red_flags.append("sensationalism")'
)

# Remove clickbait_capitalization flag if it was added (we dropped has_caps_title)
new_src = new_src.replace(
    '        if has_caps_title:\n            red_flags.append("clickbait_capitalization")\n', ''
)

# Change decision threshold from 0.55 to 0.52
new_src = new_src.replace(
    'label = "Real News" if real_prob >= 0.55 else "Fake News"',
    'label = "Real News" if real_prob >= 0.52 else "Fake News"'
)

with open('routes.py', 'w', encoding='utf-8') as f:
    f.write(new_src)

print("routes.py patched successfully.")
print(f"New file size: {len(new_src)} bytes")
