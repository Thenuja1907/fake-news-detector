from pymongo import MongoClient

# Connection string to your MongoDB Atlas
MONGO_URI = "mongodb+srv://manivannanthenuja_db_user:Thenuja123M@cluster0.jlu2yik.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client["fake_news_db"]

# 1. Collection for storing analyzed news (History)
analysis_collection = db["analyses"]

# 2. Collection for the Source Database (The missing one!)
source_collection = db["sources"]

# 3. Collection for User Management
user_collection = db["users"]

print("✓ MongoDB Collections Initialized")