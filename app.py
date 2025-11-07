# app.py - production-ready chatbot backend

import os
import datetime
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==== SETUP ====
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==== ERROR HANDLER ====
@app.errorhandler(Exception)
def handle_exception(e):
    """Ensure all errors return JSON instead of HTML."""
    logger.exception("Unhandled exception:")
    return jsonify({"error": "internal_server_error", "message": str(e)}), 500


# ==== IMPORT CONFIG, ROUTES ====
from config import limiter
limiter.init_app(app)
import routes  # noqa: E402 (import after app is created)

# ==== MAIN ====
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"✅ Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)
