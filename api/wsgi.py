# wsgi.py
from app import app  # Make sure this points to your Flask app instance
from whitenoise import WhiteNoise
import os

# Optional: explicitly define static and template folders
app.static_folder = os.path.join(os.path.dirname(__file__), 'static')
app.template_folder = os.path.join(os.path.dirname(__file__), 'templates')

# Wrap app with WhiteNoise
app.wsgi_app = WhiteNoise(app.wsgi_app, root=app.static_folder)

# Required for WSGI servers
application = app
