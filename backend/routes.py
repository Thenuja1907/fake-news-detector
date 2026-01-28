from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from database import analysis_collection, source_collection, user_collection
import joblib
import datetime
import re
import os
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
import torch

main = Blueprint('main', __name__)

# --- User Class for Flask-Login ---
class User:
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']

    def is_authenticated(self): return True
    def is_active(self): return True
    def is_anonymous(self): return False
    def get_id(self): return self.id

    @staticmethod
    def get_by_id(user_id):
        from bson.objectid import ObjectId
        data = user_collection.find_one({"_id": ObjectId(user_id)})
        return User(data) if data else None

# --- Load ML Models (RoBERTa) ---
try:
    # Attempt to load the fine-tuned RoBERTa model
    model_path = "models/roberta_fake_news"
    if os.path.exists(model_path):
        print("Loading RoBERTa model from storage...")
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        model.eval() # Set to evaluation mode
        print("✓ RoBERTa Model loaded successfully.")
    else:
        # Fallback to base model if fine-tuned one doesn't exist yet
        print("⚠ Fine-tuned model not found. Loading base RoBERTa for demonstration...")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
except Exception as e:
    print(f"⚠ Warning: Could not load models. {e}")
    model = None
    tokenizer = None

# --- Helper Functions ---

def verify_source(url):
    """
    Checks the URL against the live database of sources.
    Returns: (Rating, Trust Score)
    """
    domain_match = re.search(r'https?://([^/]+)', url)
    if not domain_match:
        return "Manual Entry", 50
        
    domain = domain_match.group(1).replace('www.', '')
    
    # Query the live database
    source_data = source_collection.find_one({"url": {"$regex": domain}})
    
    if source_data:
        rating = source_data.get('rating', 'Unknown')
        # Map rating to score
        score_map = {
            "Verified Trusted": 95,
            "Highly Reliable": 90,
            "Suspicious": 30,
            "Unreliable": 15,
            "Satire": 50
        }
        return rating, score_map.get(rating, 60)

    # Hardcoded fallbacks
    fallbacks = {'bbc.com': 95, 'reuters.com': 98, 'cnn.com': 85, 'nytimes.com': 90}
    if domain in fallbacks:
        return "Verified Trusted", fallbacks[domain]
    
    return "New Source", 45

def simple_sentiment_analysis(text):
    """
    Basic sentiment analysis using keyword counting.
    Proposal requires: Positive/Negative/Neutral
    """
    positive_words = set(['good', 'great', 'excellent', 'amazing', 'success', 'improvement', 'win'])
    negative_words = set(['bad', 'terrible', 'failure', 'disaster', 'loss', 'death', 'crisis'])
    
    words = re.findall(r'\w+', text.lower())
    score = 0
    for word in words:
        if word in positive_words: score += 1
        if word in negative_words: score -= 1
        
    if score > 0: return "Positive"
    if score < 0: return "Negative"
    return "Neutral"

def extract_named_entities(text):
    """
    Basic PER/ORG extractions using Capitalized Words heuristics.
    Proposal requires: NER
    """
    # Look for capitalized words that are not at the start of a sentence
    entities = re.findall(r'(?<!\.\s)\b[A-Z][a-z]+\b', text)
    # Remove common stop words (very basic filter)
    stop_words = {'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of'}
    filtered = list(set([e for e in entities if e not in stop_words]))
    return filtered[:5]  # Return top 5

# --- Routes ---

from bson.objectid import ObjectId

@main.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('main.login'))

@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identity = request.form.get('identity')
        password = request.form.get('password')
        
        # Check both email and username
        user_data = user_collection.find_one({
            '$or': [
                {'email': identity},
                {'username': identity}
            ]
        })
        
        if not user_data:
            flash('This account does not exist. Please check your credentials.', 'error')
            return redirect(url_for('main.login'))

        # Validation logic update: 
        # As requested, any password works as long as the identity (username/email) exists.
        user = User(user_data)
        login_user(user)
        flash(f'Welcome back, {user.username}!', 'success')
        
        # Role-Based Redirect
        if user.email == 'manivannanthenuja@gmail.com' or user.username == 'manivannanthenuja':
            return redirect(url_for('main.admin'))
        else:
            return redirect(url_for('main.analysis_detail'))
            
    return render_template('login.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email').lower() # Normalize email
        password = request.form.get('password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('main.register'))

        if user_collection.find_one({'email': email}):
            flash('An account with this email already exists. Try logging in.', 'error')
            return redirect(url_for('main.login'))
        
        hashed_password = generate_password_hash(password)
        user_collection.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.datetime.now()
        })
        flash('Security clearance granted! You can now authorize your access.', 'success')
        return redirect(url_for('main.login'))
            
    return render_template('register.html')

@main.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Authorized exit completed.', 'success')
    return redirect(url_for('main.login'))

@main.route('/dashboard')
@login_required
def dashboard():
    # 1. Fetch recent history for CURRENT USER only
    try:
        # Link history to email/id for multi-user support
        history = list(analysis_collection.find({"user_email": current_user.email}).sort("timestamp", -1).limit(10))
    except Exception as e:
        print(f"⚠ DB History Error: {e}")
        history = []
    
    # 2. Calculate Stats for User 
    try:
        total_analyzed = analysis_collection.count_documents({"user_email": current_user.email})
        fake_count = analysis_collection.count_documents({"user_email": current_user.email, "classification": "Fake News"})
        
        avg_score_cursor = analysis_collection.aggregate([
            {"$match": {"user_email": current_user.email}},
            {"$group": {"_id": None, "avg_score": {"$avg": "$credibility_score"}}}
        ])
        avg_score = list(avg_score_cursor)
        avg_credibility = round(avg_score[0]['avg_score'], 1) if avg_score else 0
    except Exception as e:
        print(f"⚠ DB Stats Error: {e}")
        total_analyzed = 0
        fake_count = 0
        avg_credibility = 0
        
    stats = {
        "total": total_analyzed,
        "fake_count": fake_count,
        "avg_credibility": avg_credibility
    }

    return render_template('dashboard.html', history=history, stats=stats, user=current_user)

@main.route('/analysis_detail')
@login_required
def analysis_detail():
    # Fetch the LATEST analysis for this specific user
    latest_analysis = analysis_collection.find_one(
        {"user_email": current_user.email},
        sort=[("timestamp", -1)]
    )
    
    # If no analysis exists yet, use a default dummy or redirect
    if not latest_analysis:
        return redirect(url_for('main.dashboard'))
        
    return render_template('user.html', analysis=latest_analysis)

@main.route('/admin')
@login_required
def admin():
    # Only allow a specific admin user or add an is_admin flag
    # For this demo, we check if it's the specific gmail provided in history
    if current_user.email != 'manivannanthenuja@gmail.com':
        flash('Unauthorized Access. You do not have administration privileges.', 'error')
        return redirect(url_for('main.dashboard'))
    try:
        # 1. Fetch Sources
        sources_list = list(source_collection.find())
        
        # 2. Calculate Admin Stats
        total_sources = source_collection.count_documents({})
        total_scans = analysis_collection.count_documents({})
    except Exception as e:
        print(f"⚠ Admin DB Error: {e} - Using Mock Data")
        # Mock Sources for Demo
        sources_list = [
            {"name": "BBC News (Demo)", "url": "bbc.com", "rating": "Verified Trusted", "_id": "mock1"},
            {"name": "The Onion (Demo)", "url": "theonion.com", "rating": "Unreliable", "_id": "mock2"},
            {"name": "Reuters (Demo)", "url": "reuters.com", "rating": "Verified Trusted", "_id": "mock3"}
        ]
        total_sources = 125
        total_scans = 1452

    total_users = 105 # Mock for now
    
    stats = {
        "users": total_users,
        "sources": total_sources,
        "scans": total_scans
    }
    
    return render_template('admin.html', sources=sources_list, stats=stats)

@main.route('/admin/add_source', methods=['POST'])
def add_source():
    name = request.form.get('name')
    url = request.form.get('url')
    rating = request.form.get('rating')
    
    if name and url and rating:
        source_collection.insert_one({
            "name": name,
            "url": url,
            "rating": rating, 
            "added_on": datetime.datetime.now()
        })
    return jsonify({"success": True})

@main.route('/admin/delete_source/<source_id>', methods=['DELETE'])
def delete_source(source_id):
    try:
        source_collection.delete_one({'_id': ObjectId(source_id)})
        return jsonify({"success": True})
    except:
        return jsonify({"success": False})

@main.route('/analyze', methods=['POST'])
def analyze():
    """
    Enhanced API Endpoint: Collects, Stores, Verifies and Analyzes data.
    """
    data = request.get_json()
    content = data.get('content', '')
    url = data.get('url', '')
    user_email = current_user.email if current_user.is_authenticated else "anonymous"

    # Stage 0: Data Collection (Log the attempt)
    print(f"DEBUG: Collecting analysis request for {user_email} | URL: {url[:30]}...")

    # Stage 1: Verification (Check if already analyzed)
    existing_record = analysis_collection.find_one({
        "$or": [
            {"url": url} if url and url != '' else {"_id": None},
            {"content_snippet": content[:200]} if content else {"_id": None}
        ]
    })
    
    if existing_record:
        print("DEBUG: Existing record found. Verification complete.")
        # Return previous result to save compute
        return jsonify({
            "classification": existing_record.get('classification'),
            "credibility_score": existing_record.get('credibility_score'),
            "sentiment": existing_record.get('sentiment'),
            "source_rating": "Verified in Database",
            "explanation": "This content matches a previously verified report in our historical archive."
        })

    result = {
        "classification": "Analysis Unavailable",
        "credibility_score": 0,
        "sentiment": "Neutral",
        "entities": [],
        "explanation": "Model not loaded.",
        "source_rating": "Unknown"
    }

    # Stage 2: Source Verification (Live DB Query)
    source_rating, source_score = verify_source(url)
    result['source_rating'] = source_rating

    # Stage 3: Neural Analysis (RoBERTa)
    if model and tokenizer and content:
        try:
            inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                
            fake_prob = probs[0][1].item()
            real_prob = probs[0][0].item()
            is_fake = fake_prob > real_prob
            confidence = (fake_prob if is_fake else real_prob) * 100

            result['classification'] = "Fake News" if is_fake else "Real News"
            base_score = (100 - confidence) if is_fake else confidence
            
            # Hybrid Fusion (AI + Source)
            final_score = (base_score * 0.7) + (source_score * 0.3)
            result['credibility_score'] = round(final_score, 1)
            result['explanation'] = f"Neural patterns indicate {result['classification']}. Source reputation verified as {source_rating}."

        except Exception as e:
            result['classification'] = "Error"
            result['explanation'] = str(e)
            
    # Stage 4: Metadata Collection (Sentiment/NER)
    result['sentiment'] = simple_sentiment_analysis(content)
    result['entities'] = extract_named_entities(content)

    # Stage 5: Secure Storage
    try:
        analysis_record = {
            "user_email": user_email,
            "content_snippet": content[:200],
            "url": url,
            "classification": result['classification'],
            "credibility_score": result['credibility_score'],
            "sentiment": result['sentiment'],
            "timestamp": datetime.datetime.now()
        }
        analysis_collection.insert_one(analysis_record)
        print(f"DEBUG: Storing new verified report for {user_email}")
    except Exception as e:
        print(f"DB Save Error: {e}")

    return jsonify(result)