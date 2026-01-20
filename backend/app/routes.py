from flask import Blueprint, render_template, request, jsonify

# This organizes our routes
main = Blueprint('main', __name__)

@main.route('/')
def index():
    # This will look for index.html in the templates folder
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    news_text = data.get('text', '')
    
    # Mock AI Logic
    # In a real app, you would call your ML model here
    clickbait_words = ['shocking', 'unbelievable', 'secret', 'won\'t believe']
    is_fake = any(word in news_text.lower() for word in clickbait_words)
    
    result = {
        "classification": "Fake News" if is_fake else "Real News",
        "score": 0.92 if is_fake else 0.12,
        "explanation": "High clickbait word density detected." if is_fake else "Content appears standard."
    }
    return jsonify(result)