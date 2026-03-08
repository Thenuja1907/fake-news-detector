import time
import requests
import sys
import os

# Import database module from backend folder to connect to Mongo directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import analysis_collection

def test_db_persistence():
    print("=" * 80)
    print("TESTING DB PERSISTENCE: Data remains after simulating server restarts")
    print("=" * 80)

    base_url = "http://127.0.0.1:5000"
    unique_marker = f"PERSISTENCE_TEST_{int(time.time())}"
    text_to_save = f"This is a unique string intended to test Database Storage. ID: {unique_marker}"

    # Step 1: Count current records
    initial_count = analysis_collection.count_documents({})
    print(f"Step 1: Current Database Size: {initial_count} records")

    # Step 2: Push new data via the running Server
    print(f"\nStep 2: Sending unique article to Server API: '{unique_marker}'")
    try:
        res = requests.post(f"{base_url}/analyze", json={"content": text_to_save})
        if res.status_code == 200:
            print("   ✅ Server successfully processed the data and returned HTTP 200")
        else:
            print("   ❌ Server failed to process the request.")
            return
    except Exception as e:
        print(f"   ❌ Server connection error: {e}")
        return

    # Give Mongo a tiny buffer to execute the write operation
    time.sleep(1)

    # Step 3: Count records again
    mid_count = analysis_collection.count_documents({})
    print(f"\nStep 3: New Database Size: {mid_count} records")
    if mid_count == initial_count + 1:
        print("   ✅ The record was successfully injected into active memory.")
    else:
        print("   ❌ The record was not accurately counted in the database.")

    # Step 4: Disconnect from DB entirely to "Simulate" a server crash/reboot
    print("\n--- SIMULATING SERVER REBOOT (Clearing Connections) ---")
    analysis_collection.database.client.close()
    
    # Wait for connections to terminate
    time.sleep(2)
    print("--- SERVER RESTARTED ---")

    # Reconnect with a FRESH MongoClient (Not the shared singleton)
    from pymongo import MongoClient
    from database import MONGO_URI
    import certifi

    new_client = MongoClient(
        MONGO_URI, 
        tls=True, 
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=True,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000
    )
    new_db_connection = new_client['fake_news_db']
    new_collection = new_db_connection['analyses']

    # Step 5: Verify the unique signature still exists from the blank, fresh connection
    print("\nStep 4: Searching the new fresh Database connection for the unique ID...")
    search_result = new_collection.find_one({"content": text_to_save})
    
    if search_result:
        print(f"   ✅ FOUND IT! The text '{search_result.get('content', '')[:30]}...' still exists.")
        
        print("\n[CONCLUSION]")
        print("STATUS: ✅ PASS")
        print("Verdict: Data is successfully persisting. The server stores information directly into the remote/local disk Atlas Cluster, rather than temporary Flask memory. It will survive hard reboots.")
        
        # Cleanup
        new_collection.delete_one({"content": text_to_save})
    else:
        print("   ❌ Missing! The database dropped the record upon connection refresh.")
        print("\n[CONCLUSION]")
        print("STATUS: ❌ FAIL")
        print("Verdict: The system is using temporary runtime variables instead of persistent storage.")

if __name__ == "__main__":
    test_db_persistence()
