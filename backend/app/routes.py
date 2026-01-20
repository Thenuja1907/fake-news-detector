from flask import Blueprint, render_template, request, jsonify
from .database import analysis_collection
import datetime
import random

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    content = data.get('content', '')

    # AI Simulation Logic
    suspicious_words = ['shocking', 'unbelievable', 'secret', 'miracle']
    suspicious_count = sum(1 for word in suspicious_words if word in content.lower())
    
    score = max(20, 95 - (suspicious_count * 15) - random.randint(0, 10))
    classification = "Real" if score > 70 else "Misleading" if score > 40 else "Fake"
    
    result = {
        "content": content[:100] + "...",
        "classification": classification,
        "credibilityScore": int(score),
        "sentiment": "Negative" if suspicious_count > 1 else "Positive",
        "emotion": "Fear/Shock" if suspicious_count > 1 else "Neutral",
        "explanation": f"Analysis shows {classification.lower()} characteristics with {int(score)}% confidence.",
        "entities": ["Person: John Doe", "Location: New York"],
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    # Save to MongoDB
    analysis_collection.insert_one(result.copy())
    
    # Remove _id before sending to frontend
    if '_id' in result: del result['_id']
    
    return jsonify(result)

@main.route('/history', methods=['GET'])
def get_history():
    history = list(analysis_collection.find().sort("timestamp", -1).limit(10))
    for item in history:
        item['_id'] = str(item['_id'])
    return jsonify(history)