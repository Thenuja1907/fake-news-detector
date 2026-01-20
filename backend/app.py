from flask import Flask, request, jsonify

app = Flask(__name__)

# --- 1. DEFINE THE MISSING FUNCTIONS FIRST ---

def classify_text(content):
    # Simple logic: if it has "breaking" or "shocking", call it Fake
    clickbait = ["breaking", "shocking", "unbelievable", "secret"]
    if any(word in content.lower() for word in clickbait):
        return "Fake News"
    return "Real News"

def get_credibility_score(content):
    # Returns a mock score out of 100
    return 85 if len(content) > 50 else 40

def analyze_sentiment(content):
    return "Neutral"

def extract_entities(content):
    # Mock entity extraction
    return ["AI", "News Source"]

def generate_explanation(content):
    return "The content was analyzed based on keyword patterns and length."

# --- 2. THE ROUTE THAT USES THE FUNCTIONS ---

@app.route('/analyze', methods=['POST'])
def analyze_content():
    # Get the data sent from the frontend
    data = request.json
    content = data.get('content', '')

    # Perform analysis using the functions defined above
    result = {
        'classification': classify_text(content),
        'credibilityScore': get_credibility_score(content),
        'sentiment': analyze_sentiment(content),
        'entities': extract_entities(content),
        'explanation': generate_explanation(content)
    }

    return jsonify(result)

# --- 3. RUN THE APP ---

if __name__ == '__main__':
    app.run(debug=True, port=5000)