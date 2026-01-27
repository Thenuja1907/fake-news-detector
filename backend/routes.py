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
    Checks the URL against a known database of sources.
    Returns: (Rating, Trust Score)
    """
    # In a real app, this would query 'source_collection' in MongoDB
    # For now, we use a hardcoded list for demonstration
    trusted_sources = {'bbc.com': 95, 'reuters.com': 98, 'cnn.com': 85, 'nytimes.com': 90}
    unreliable_sources = {'fake-news.com': 10, 'conspiracy-daily.org': 15}
    
    domain = re.search(r'https?://([^/]+)', url)
    if domain:
        domain = domain.group(1).replace('www.', '')
        
        if domain in trusted_sources:
            return "Verified Trusted", trusted_sources[domain]
        elif domain in unreliable_sources:
            return "Unreliable", unreliable_sources[domain]
    
    return "Unknown Source", 50  # Default neutral score

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
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_data = user_collection.find_one({'email': email})
        
        if not user_data:
            flash('This account does not exist. Please check your credentials.', 'error')
            return redirect(url_for('main.login'))

        if check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            
            # Role-Based Redirect
            if user.email == 'manivannanthenuja@gmail.com':
                return redirect(url_for('main.admin'))
            else:
                return redirect(url_for('main.dashboard'))
        else:
            flash('Incorrect password. Please try again.', 'error')
            return redirect(url_for('main.login'))
            
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
    Main API Endpoint expected by the Browser Extension.
    """
    data = request.get_json()
    content = data.get('content', '')
    url = data.get('url', '')
    
    result = {
        "classification": "Analysis Unavailable",
        "credibility_score": 0,
        "sentiment": "Neutral",
        "entities": [],
        "explanation": "Model not loaded.",
        "source_rating": "Unknown"
    }

    # 1. Source Verification
    source_rating, source_score = verify_source(url)
    result['source_rating'] = source_rating

    # 2. Content Classification (RoBERTa)
    if model and tokenizer and content:
        try:
            # Tokenize
            inputs = tokenizer(
                content, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                
            # Assume 0 = Real, 1 = Fake (Alignment depends on training)
            # Let's assume the model outputs [Real_Prob, Fake_Prob]
            fake_prob = probs[0][1].item()
            real_prob = probs[0][0].item()
            
            is_fake = fake_prob > real_prob
            confidence = (fake_prob if is_fake else real_prob) * 100

            result['classification'] = "Fake News" if is_fake else "Real News"
            
            # 3. Overall Credibility Score
            # If it's Fake, credibility is low ( inverse of confidence)
            # If it's Real, credibility is high (confidence)
            base_score = (100 - confidence) if is_fake else confidence
            
            # Fuse with source score
            final_score = (base_score * 0.7) + (source_score * 0.3)
            result['credibility_score'] = round(final_score, 1)
            
            result['explanation'] = (
                f"RoBERTa analysis indicates {result['classification']} with {round(confidence, 1)}% confidence. "
                f"Source is rated as {source_rating}."
            )

        except Exception as e:
            print(f"Error during prediction: {e}")
            result['classification'] = "Error"
            result['explanation'] = str(e)
            
    # 4. Sentiment & NER
    result['sentiment'] = simple_sentiment_analysis(content)
    result['entities'] = extract_named_entities(content)

    # 5. Save to Database
    try:
        user_email = current_user.email if current_user.is_authenticated else "anonymous"
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
    except Exception as e:
        print(f"DB Save Error: {e}")

    return jsonify(result)