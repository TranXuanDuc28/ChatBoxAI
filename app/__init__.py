from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Đăng ký routes
    from app.routes.core import register_core_routes
    from app.routes.management import register_management_routes
    from app.routes.testing import register_testing_routes
    from app.routes.error_handlers import register_error_handlers
    
    register_core_routes(app)
    register_management_routes(app)
    register_testing_routes(app)
    register_error_handlers(app)
    
    return app