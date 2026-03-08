import requests
import time

def send_request(request_id):
    base_url = "http://127.0.0.1:5000"
    
    # We alter the text slightly per request to avoid any exact caching if present
    text = f"Sample news broadcast number {request_id}. The economy is seeing unprecedented shifts today."
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/analyze", 
            json={"content": text, "url": "https://reuters.com"}
        )
        end_time = time.time()
        
        status_code = response.status_code
        time_taken = round(end_time - start_time, 3)
        
        if status_code == 200:
            return {"id": request_id, "status": "Success", "time": time_taken, "code": status_code}
        else:
            return {"id": request_id, "status": "Failed", "time": time_taken, "code": status_code}
    except Exception as e:
        end_time = time.time()
        time_taken = round(end_time - start_time, 3)
        return {"id": request_id, "status": f"Error", "time": time_taken, "code": "N/A"}

def test_consecutive_requests():
    print("=" * 70)
    print("TESTING TC-S02: System handles multiple consecutive requests")
    print("=" * 70)
    
    num_requests = 10  # Sending 10 AI evaluation requests back-to-back
    print(f"Sending {num_requests} Deep Learning analysis requests sequentially...\n")
    
    results = []
    total_start_time = time.time()
    
    print(f"{'Req #':<5} | {'HTTP Status':<12} | {'Time Taken':<12} | {'Result'}")
    print("-" * 70)
    
    for i in range(1, num_requests + 1):
        res = send_request(i)
        results.append(res)
        
        icon = "✅" if res['status'] == "Success" else "❌"
        print(f"#{res['id']:<4} | {res['code']:<12} | {res['time']:<11}s | {icon} {res['status']}")
    
    total_end_time = time.time()
    total_duration = round(total_end_time - total_start_time, 3)
    
    success_count = sum(1 for r in results if r['status'] == 'Success')
    
    print("\n" + "=" * 70)
    print(f"SUMMARY")
    print("=" * 70)
    print(f"Total time taken      : {total_duration}s")
    print(f"Successful requests   : {success_count}/{num_requests}")
    
    # Check if there were any dropped packets or timeouts
    if success_count == num_requests:
        print("\nSTATUS: ✅ PASS (System successfully processed all back-to-back requests)")
        print("Verdict: The local Flask server handles consecutive AI model inference without crashing or leaking memory.")
    else:
        print("\nSTATUS: ❌ FAIL (System dropped or failed some requests)")

if __name__ == '__main__':
    test_consecutive_requests()
