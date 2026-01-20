import sys
import os
from flask import Flask

# This forces Python to look in the current folder for routes.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from routes import main
except ImportError as e:
    print(f"Error importing routes: {e}")
    sys.exit(1)

app = Flask(__name__, 
            template_folder='app/templates', 
            static_folder='app/static')

app.register_blueprint(main)

if __name__ == "__main__":
    print("-" * 30)
    print("AI FAKE NEWS DETECTION SYSTEM")
    print("-" * 30)
    # The accuracy you achieved in your previous step
    print("✓ Model Loaded: Accuracy 60.25%") 
    print("→ Server: http://127.0.0.1:5000")
    print("-" * 30)
    app.run(debug=True, port=5000)