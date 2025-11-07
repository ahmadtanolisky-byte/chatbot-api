# app.py
from flask import jsonify, request
from config import app, db, limiter, client, index, PORT, TOP_K, CONTEXT_MESSAGES, MAX_CONTEXT_CHARS, RATE_LIMIT, logger
from models import Session, Message
from utils import save_message, get_recent_messages, trim_context_text, require_basic_auth
import datetime
from routes import *
from flask_cors import CORS



SYSTEM_PROMPT = (
    "You are a helpful assistant for the Faisal Town / Sky Marketing website. "
    "Answer using only the provided website content and conversation context. "
    "If unsure, reply politely and suggest contacting the sales team."
)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Chatbot API running"})

# ... Copy your /session, /chat, /save-chat, /admin endpoints here unchanged ...
# You can also move them into routes.py if you want.

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=PORT)
