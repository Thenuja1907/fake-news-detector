from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import certifi
import datetime

MONGO_URI = "mongodb+srv://manivannanthenuja_db_user:Thenuja123M@cluster0.jlu2yik.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(
    MONGO_URI, 
    tls=True, 
    tlsCAFile=certifi.where(),
    tlsAllowInvalidCertificates=True
)
db = client["fake_news_db"]
user_collection = db["users"]

demo_users = [
    {
        "username": "AI Observer",
        "email": "user@demo.com",
        "password": generate_password_hash("anypass"),
        "created_at": datetime.datetime.now()
    },
    {
        "username": "System Administrator",
        "email": "admin@demo.com",
        "password": generate_password_hash("anypass"),
        "created_at": datetime.datetime.now()
    }
]

for user in demo_users:
    # Update if exists, insert if not
    user_collection.update_one(
        {"email": user["email"]},
        {"$set": user},
        upsert=True
    )
    print(f"Verified demo account: {user['email']}")

print("✓ Demo seeding complete.")
