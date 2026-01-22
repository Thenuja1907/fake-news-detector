from flask import Flask
from flask_cors import CORS
from routes import main

app = Flask(__name__)
CORS(app) # Enable CORS for browser extension access

app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True, port=5000)