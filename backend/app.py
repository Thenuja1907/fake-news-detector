from flask import Flask
from flask_cors import CORS
from routes import main

from flask_login import LoginManager
from routes import main, User # Import User class

app = Flask(__name__, template_folder='app/templates')
app.secret_key = 'super_secret_key_for_demo' # Change this in production!

# --- Login Manager Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'main.login'

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)

CORS(app) # Enable CORS for browser extension access

app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True, port=5000)