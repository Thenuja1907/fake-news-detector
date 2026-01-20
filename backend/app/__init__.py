from flask import Flask

def create_app():
    app = Flask(__name__)

    # Register the routes from routes.py
    from .routes import main
    app.register_blueprint(main)

    return app