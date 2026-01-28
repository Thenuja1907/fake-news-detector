from pymongo import MongoClient

import certifi
import datetime
from werkzeug.security import generate_password_hash

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
import json
import os

class MockCollection:
    def __init__(self, data=None, collection_name="default"):
        import secrets
        # Use absolute path to ensure persistence in the correct directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(base_dir, f"mock_{collection_name}.json")
        print(f"DEBUG: Initializing MockCollection {collection_name} at {self.filename}")
        
        self.data = []
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    content = f.read().strip()
                    if content:
                        loaded_data = json.loads(content)
                        # Convert ISO strings back to datetime objects for known date fields
                        datetime_fields = ['timestamp', 'last_login', 'created_at']
                        for doc in loaded_data:
                            for field in datetime_fields:
                                if field in doc and isinstance(doc[field], str):
                                    try:
                                        doc[field] = datetime.datetime.fromisoformat(doc[field])
                                    except: pass
                        self.data = loaded_data
                        print(f"DEBUG: Loaded {len(self.data)} records from {self.filename}")
                    else:
                        print(f"DEBUG: File {self.filename} is empty, using default data")
                        self.data = data or []
            except Exception as e:
                print(f"DEBUG Error loading mock {collection_name}: {e}")
                self.data = data or []
        else:
            print(f"DEBUG: No existing file for {collection_name}, using default data")
            self.data = data or []
            
        # Ensure default data is present if data is empty
        if not self.data and data:
            self.data = data
            
        for doc in self.data:
            if '_id' not in doc:
                doc['_id'] = secrets.token_hex(12)
        
        self._save()

    def _save(self):
        try:
            def default_serializer(obj):
                if isinstance(obj, datetime.datetime):
                    return obj.isoformat()
                return str(obj)
            
            with open(self.filename, 'w') as f:
                json.dump(self.data, f, default=default_serializer, indent=4)
            print(f"DEBUG: Saved {len(self.data)} records to {self.filename}")
        except Exception as e:
            print(f"DEBUG Error saving mock {self.filename}: {e}")

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
        if sort: return results[-1]
        return results[0]

    def insert_one(self, doc):
        if '_id' not in doc:
            import secrets
            doc['_id'] = secrets.token_hex(12)
        self.data.append(doc)
        self._save()
        return type('OBJ', (), {'inserted_id': doc['_id']})
    
    def find(self, query=None, sort=None, limit=None):
        if not query: return MockCursor(self.data)
        filtered = [item for item in self.data if all(item.get(k) == v for k, v in query.items())]
        return MockCursor(filtered) 
    
    def count_documents(self, query):
        if not query: return len(self.data)
        return len([item for item in self.data if all(item.get(k) == v for k, v in query.items())])

    def update_one(self, query, update):
        target = self.find_one(query)
        if target and '$set' in update:
            for k, v in update['$set'].items():
                target[k] = v
            self._save()
        return type('OBJ', (), {'modified_count': 1 if target else 0})

    def aggregate(self, pipeline):
        match_query = {}
        for stage in pipeline:
            if '$match' in stage: match_query = stage['$match']
        
        filtered = [item for item in self.data if all(item.get(k) == v for k, v in match_query.items())]
        total_score = sum(item.get('credibility_score', 0) for item in filtered)
        avg_score = total_score / len(filtered) if filtered else 0
        return [{"avg_score": avg_score}]

class MockCursor:
    def __init__(self, data):
        self.data = data
    def sort(self, *args, **kwargs): return self
    def limit(self, *args, **kwargs): return self
    def __iter__(self): return iter(self.data)
    def __list__(self): return self.data

mock_users = [
    {
        "username": "manivannanthenuja",
        "email": "manivannanthenuja@gmail.com",
        "password": generate_password_hash("anypass")
    },
    {
        "username": "Demo User",
        "email": "user@demo.com",
        "password": generate_password_hash("anypass")
    },
    {
        "username": "John Doe",
        "email": "john@example.com",
        "password": generate_password_hash("password")
    },
    {
        "username": "Jane Smith",
        "email": "jane@example.com",
        "password": generate_password_hash("password")
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
            "user_email": "user@demo.com",
            "content_snippet": "New study suggests that drinking 5 cups of coffee daily grants immortality.",
            "url": "https://daily-science-buzz.com/immortality-coffee",
            "classification": "Fake News",
            "credibility_score": 12.5,
            "sentiment": "Excitement",
            "timestamp": datetime.datetime.now()
        }
    ], collection_name="analyses")
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
    ], collection_name="sources")
    user_collection = MockCollection(mock_users, collection_name="users")

