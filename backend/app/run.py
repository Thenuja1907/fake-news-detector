from app import create_app

# This creates the Flask app using the factory we defined in app/__init__.py
app = create_app()

if __name__ == "__main__":
    # Start the server on port 5000
    app.run(debug=True, port=5000)