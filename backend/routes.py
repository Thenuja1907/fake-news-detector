from flask import Blueprint, render_template, request, jsonify
from database import analysis_collection, source_collection
import joblib
import datetime
import re
import os
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
import torch

main = Blueprint('main', __name__)

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

@main.route('/')
def index():
    return render_template('dashboard.html')

@main.route('/dashboard')
def dashboard():
    # Fetch recent history from DB
    history = list(analysis_collection.find().sort("timestamp", -1).limit(10))
    return render_template('dashboard.html', history=history)

@main.route('/admin')
def admin():
    sources_list = list(source_collection.find())
    return render_template('admin.html', sources=sources_list)

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
        analysis_record = {
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