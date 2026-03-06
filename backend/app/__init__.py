from flask import Flask

def create_app():
    from flask_cors import CORS
    from flask_login import LoginManager
    from database import user_collection
    from bson.objectid import ObjectId

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-secret-key-premium-123'
    # Note: Using * for dev to allow Chrome Extension content scripts (Manifest V3) to hit the endpoint.
    CORS(app, resources={r"/*": {"origins": "*"}}) 

    login_manager = LoginManager()
    login_manager.login_view = 'main.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        from routes import User
        return User.get_by_id(user_id)

    @app.after_request
    def set_no_cache(response):
        """Prevent browser from caching any page — forces re-authentication on every visit."""
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    # Register the routes from routes.py
    from routes import main
    app.register_blueprint(main)

    return app