# Import Flask class
from flask import Flask

# Create Flask application instance
app = Flask(__name__)

# Define a route (URL path) and a view function
@app.route('/')  # Route for homepage

def home():
    return "Hello! 👋 Welcome to my first Flask App!"

# Another route
@app.route('/about')
def about():
    return "This is the About Page 🚀"

# Run the app (start the Flask development server)
if __name__ == "__main__":
    app.run(debug=True)