from pymongo import MongoClient

# Replace with your actual MongoDB URI if using MongoDB Atlas
client = MongoClient("mongodb+srv://manivannanthenuja_db_user:Thenuja123M@cluster0.jlu2yik.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["fake_news_db"]

# This is the variable your routes.py is looking for
analysis_collection = db["analyses"]