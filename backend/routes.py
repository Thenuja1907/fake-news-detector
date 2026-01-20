from flask import Blueprint, render_template, request, jsonify
from database import analysis_collection, source_collection
import joblib
import datetime
import os

main = Blueprint('main', __name__)

# 1. Load the AI Model and Vectorizer we just created
MODEL_PATH = "models/fake_news_model.pkl"
VECTOR_PATH = "models/tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTOR_PATH):
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTOR_PATH)
    print("✓ AI Brain loaded into Web Server")
else:
    print("Error: AI Model files not found. Run train_model.py first.")

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/admin')
def admin():
    # Calculate stats for the Admin Dashboard UI (Image 3)
    total_analyses = analysis_collection.count_documents({})
    real_count = analysis_collection.count_documents({"verdict": "Real News"})
    fake_count = analysis_collection.count_documents({"verdict": "Fake News"})
    sources = list(source_collection.find())
    
    return render_template('admin.html', 
                           total=total_analyses, 
                           real=real_count, 
                           fake=fake_count, 
                           sources=sources)

@main.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    content = data.get('content', '')
    url = data.get('url', '')
    
    # 2. Use the AI Brain to predict
    vec_text = tfidf.transform([content])
    prediction = model.predict(vec_text)
    
    # Map 0/1 back to Real/Fake labels
    verdict = "Real News" if prediction[0] == 1 else "Fake News"
    confidence = "60.25%" # From your training result

    # 3. Save to History (Image 2) & MongoDB
    record = {
        "content": content[:100] + "...",
        "url": url,
        "verdict": verdict,
        "confidence": confidence,
        "timestamp": datetime.datetime.now()
    }
    analysis_collection.insert_one(record)

    return jsonify({
        "verdict": verdict, 
        "confidence": confidence,
        "explanation": f"Based on linguistic patterns, this content matches {verdict.lower()} signatures."
    })