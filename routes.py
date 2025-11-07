# routes.py - chatbot routes

from flask import request, jsonify
from config import app, limiter, client, index, logger
from config import TOP_K, CONTEXT_MESSAGES, MAX_CONTEXT_CHARS
from config import SYSTEM_PROMPT, is_short_query
from models import Session, Message
from utils import save_message, get_recent_messages, trim_context_text, require_basic_auth
import json

@app.route("/chat", methods=["POST"])
@limiter.limit("10/minute")
def chat():
    """
    Handles chatbot queries from frontend.
    """
    try:
        data = request.get_json(force=True)
        session_id = data.get("session_id")
        question = data.get("question")

        if not session_id or not question:
            return jsonify({"error": "missing_fields"}), 400

        # Handle short queries quickly
        if is_short_query(question):
            logger.info(f"Short query detected: {question}")

        # Retrieve previous messages
        history = get_recent_messages(session_id, CONTEXT_MESSAGES)
        context_text = trim_context_text(history, MAX_CONTEXT_CHARS)

        # Generate answer using OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )

        answer = completion.choices[0].message.content.strip()

        # Save chat
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)

        return jsonify({"answer": answer}), 200

    except Exception as e:
        logger.exception("Error in /chat route:")
        return jsonify({"error": "internal_error", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Chatbot backend running"}), 200
