# Replace 'NewsProject123' with the from pymongo import MongoClient
from pymongo import MongoClient
import urllib.parse
import sys

# 1. Enter your credentials here
username = "manivannanthenuja_db_user"
password = "Thenuja123M"  # If you changed this in Atlas, update it here
cluster = "cluster0.jlu2yik.mongodb.net"

# 2. This part "cleans" the password so special characters don't cause errors
safe_password = urllib.parse.quote_plus(password)

# 3. The final Connection String
MONGO_URI = f"mongodb+srv://{username}:{safe_password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"

try:
    # Connect to MongoDB
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    
    # Trigger a connection check (The "Ping")
    client.admin.command('ping')
    print("✅ SUCCESS: Connected to MongoDB Atlas!")
    
    # Define the Database name
    db = client['fake_news_system']

    # Define the Collections (Your Schemas)
    users_collection = db['users']
    analysis_collection = db['analysis']
    sources_collection = db['sources']

except Exception as e:
    print("\n❌ CONNECTION ERROR:")
    print(f"Error Details: {e}")
    print("\n--- QUICK FIX STEPS ---")
    print("1. Check if your Password is correct in the code above.")
    print("2. In MongoDB Atlas, go to 'Network Access' and click 'Allow Access From Anywhere'.")
    print("3. In MongoDB Atlas, go to 'Database Access' and make sure the user has 'Read and Write' roles.")
    sys.exit(1)