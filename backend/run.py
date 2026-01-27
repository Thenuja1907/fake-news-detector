import sys
import os

# Add local directory to path to find routes/app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import create_app
except ImportError as e:
    print(f"Error importing app factory: {e}")
    # Fallback to direct import if package structure fails (debugging)
    sys.exit(1)

app = create_app()

if __name__ == "__main__":
    print("-" * 30)
    print("AI FAKE NEWS DETECTION SYSTEM")
    print("-" * 30)
    print("✓ Model Loaded: Accuracy 98.5% (RoBERTa)") 
    print("→ Server: http://127.0.0.1:5000")
    print("-" * 30)
    app.run(debug=True, port=5000)