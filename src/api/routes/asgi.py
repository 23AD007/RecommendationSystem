from asgiref.wsgi import WsgiToAsgi
from app import app   # this imports your Flask app instance

asgi_app = WsgiToAsgi(app)
