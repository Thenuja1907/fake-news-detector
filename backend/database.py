from pymongo import MongoClient

import certifi
import datetime

# Connection string to your MongoDB Atlas
MONGO_URI = "mongodb+srv://manivannanthenuja_db_user:Thenuja123M@cluster0.jlu2yik.mongodb.net/?retryWrites=true&w=majority"

# Bypass SSL verification for development environment issues
client = MongoClient(MONGO_URI, 
                     tls=True, 
                     tlsAllowInvalidCertificates=True,
                     serverSelectionTimeoutMS=5000) 

db = client["fake_news_db"]

# --- MOCK FALLBACK SYSTEM ---
# This ensures the app works even if MongoDB Atlas is inaccessible
class MockCollection:
    def __init__(self, data=None):
        self.data = data or []
    def find_one(self, query, sort=None):
        # Handle $or separately
        if '$or' in query:
            for condition in query['$or']:
                res = self.find_one(condition)
                if res: return res
            return None

        # Standard equality check (sorted or not)
        results = []
        for item in self.data:
            match = True
            for k, v in query.items():
                item_val = item.get(k)
                if isinstance(v, dict) and '$regex' in v:
                    if not item_val or v['$regex'] not in str(item_val):
                        match = False
                        break
                else:
                    target_val = str(item_val) if k == '_id' else item_val
                    comp_val = str(v) if k == '_id' else v
                    if target_val != comp_val:
                        match = False
                        break
            if match: results.append(item)
        
        if not results: return None
        # Sort by timestamp if possible
        if sort:
            # Basic dummy sort to pick the 'latest' index-wise for mock
            return results[-1]
        return results[0]

    def insert_one(self, doc):
        if '_id' not in doc:
            import secrets
            doc['_id'] = secrets.token_hex(12) # 24 characters total
        self.data.append(doc)
        return type('OBJ', (), {'inserted_id': doc['_id']})
    
    def find(self, query=None, sort=None, limit=None):
        if not query: 
            return MockCursor(self.data)
        # Simple filter
        filtered = [item for item in self.data if all(item.get(k) == v for k, v in query.items())]
        return MockCursor(filtered) 
    
    def count_documents(self, query):
        if not query: return len(self.data)
        return len([item for item in self.data if all(item.get(k) == v for k, v in query.items())])

class MockCursor:
    def __init__(self, data):
        self.data = data
    def sort(self, *args, **kwargs): return self
    def limit(self, *args, **kwargs): return self
    def __iter__(self): return iter(self.data)
    def __list__(self): return self.data

# Initialize default mock data for Demo
# 24-char hex IDs for compatibility with ObjectId()
mock_users = [
    {
        "_id": "00000000000000000000000a",
        "username": "Admin Guardian",
        "email": "manivannanthenuja@gmail.com",
        "password": "pbkdf2:sha256:600000$mock$4878a1bc1b6911f44a4282361405e3532cc2942f6d53b9f4851221f450c2664b" # 'password'
    },
    {
        "_id": "00000000000000000000000b",
        "username": "Demo User",
        "email": "user@truth.guardian",
        "password": "pbkdf2:sha256:600000$mock$4878a1bc1b6911f44a4282361405e3532cc2942f6d53b9f4851221f450c2664b" # 'password'
    }
]


use_fallback = False
try:
    client.admin.command('ping')
    print("✓ MongoDB Connection Verified")
    analysis_collection = db["analyses"]
    source_collection = db["sources"]
    user_collection = db["users"]
except Exception as e:
    print(f"⚠ MongoDB Connection Failed: {e}")
    print("➡ Switching to Mock Local Database for demonstration...")
    use_fallback = True
    analysis_collection = MockCollection([
        {
            "_id": "00000000000000000000000c",
            "user_email": "user@truth.guardian",
            "content_snippet": "New study suggests that drinking 5 cups of coffee daily grants immortality.",
            "url": "https://daily-science-buzz.com/immortality-coffee",
            "classification": "Fake News",
            "credibility_score": 12.5,
            "sentiment": "Excitement",
            "timestamp": datetime.datetime.now()
        }
    ])
    source_collection = MockCollection([
        {"name": "BBC News", "url": "bbc.com", "rating": "Verified Trusted"},
        {"name": "Reuters", "url": "reuters.com", "rating": "Verified Trusted"},
        {"name": "Associated Press", "url": "apnews.com", "rating": "Verified Trusted"},
        {"name": "The New York Times", "url": "nytimes.com", "rating": "Verified Trusted"},
        {"name": "The Guardian", "url": "theguardian.com", "rating": "Verified Trusted"},
        {"name": "HealthLine Verified", "url": "healthline.com", "rating": "Highly Reliable"},
        {"name": "Fake News Network", "url": "fake-news-website.com", "rating": "Unreliable"},
        {"name": "Conspiracy Blog", "url": "conspiracy-blog.com", "rating": "Unreliable"},
        {"name": "Fake News Blog", "url": "fake-news-blog.com", "rating": "Unreliable"},
        {"name": "Misinfo Today", "url": "misinfo-today.org", "rating": "Unreliable"}
    ])
    user_collection = MockCollection(mock_users)

