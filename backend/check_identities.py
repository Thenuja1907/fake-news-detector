from pymongo import MongoClient
import certifi

MONGO_URI = "mongodb+srv://manivannanthenuja_db_user:Thenuja123M@cluster0.jlu2yik.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client["fake_news_db"]

print("--- Unique user_emails in analyses ---")
emails = db["analyses"].distinct("user_email")
for e in emails[:10]:
    print(f"Email: {e}")

print("\n--- Unique emails in users collection ---")
u_emails = db["users"].distinct("email")
for e in u_emails[:10]:
    print(f"User: {e}")
