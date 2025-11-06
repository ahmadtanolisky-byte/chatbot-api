# app.py
import os
import datetime
import json
from functools import wraps

from flask import Flask, request, jsonify, abort, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ---------- Configuration ----------
PORT = int(os.getenv("PORT", 5000))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chats.db")  # override with Postgres URL in production
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")  # set to https://your-website.com in prod
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")
TOP_K = int(os.getenv("TOP_K", 6))  # number of Pinecone chunks to retrieve
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", 6))  # last N messages included for context
RATE_LIMIT = os.getenv("RATE_LIMIT", "60 per minute")  # tune as needed

# Validate essential env vars
if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise RuntimeError("OPENAI_API_KEY, PINECONE_API_KEY and PINECONE_INDEX_NAME must be set as environment variables")

# ---------- Flask + Extensions ----------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

db = SQLAlchemy(app)

# Flask-Limiter (v3+ initialization)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT],
)
limiter.init_app(app)

# ---------- OpenAI & Pinecone Clients ----------
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print("❌ OpenAI client init error:", e)
    raise

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print("❌ Pinecone client/init error:", e)
    raise

# ---------- Models ----------
class Session(db.Model):
    __tablename__ = "sessions"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(128), unique=True, nullable=False)
    name = db.Column(db.String(256))
    phone = db.Column(db.String(64))
    page = db.Column(db.String(1024))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Message(db.Model):
    __tablename__ = "messages"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(128), db.ForeignKey("sessions.session_id"), index=True)
    role = db.Column(db.String(32))  # 'user' or 'assistant'
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Create tables if not exist
with app.app_context():
    db.create_all()

# ---------- Utilities ----------
def require_basic_auth(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != ADMIN_USER or auth.password != ADMIN_PASS:
            return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})
        return func(*args, **kwargs)
    return wrapped

def get_recent_messages(session_id, limit=CONTEXT_MESSAGES):
    msgs = Message.query.filter_by(session_id=session_id).order_by(Message.id.desc()).limit(limit).all()
    # return chronological order oldest -> newest
    return list(reversed([{"role": m.role, "content": m.content} for m in msgs]))

def save_message(session_id, role, content):
    try:
        m = Message(session_id=session_id, role=role, content=content)
        db.session.add(m)
        db.session.commit()
    except Exception as e:
        # logging to console (Render logs)
        print("❌ Failed to save message:", e)

# Strict assistant prompt (will be combined with retrieved context + conversation history)
SYSTEM_PROMPT = (
    "You are a helpful assistant that must ONLY answer using the provided website content. "
    "If the answer cannot be found in the provided content, respond exactly: 'I don't know.' "
    "Be concise and return practical, factual replies. If you include a link, reference the original page URL."
)

# ---------- API Routes ----------

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Chatbot API is running."})

# Start or update session (create session record if not exists)
@app.route("/session", methods=["POST"])
@limiter.limit("10 per minute")
def start_session():
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id")
        name = data.get("name")
        phone = data.get("phone")
        page = data.get("page", "")

        if not session_id:
            return jsonify({"error": "session_id required"}), 400

        s = Session.query.filter_by(session_id=session_id).first()
        if not s:
            s = Session(session_id=session_id, name=name, phone=phone, page=page)
            db.session.add(s)
        else:
            # update info if provided
            if name: s.name = name
            if phone: s.phone = phone
            if page: s.page = page

        db.session.commit()
        return jsonify({"message": "session started", "session_id": session_id})
    except Exception as e:
        print("❌ Error in /session:", e)
        return jsonify({"error": "server error"}), 500

# Main chat endpoint: takes session_id and question, returns answer and stores messages
@app.route("/chat", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def chat():
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id")
        question = data.get("question", "").strip()
        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        if not question:
            return jsonify({"error": "question required"}), 400

        # store user message
        save_message(session_id, "user", question)

        # 1) Get embedding for the question
        try:
            q_embed = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
        except Exception as e:
            print("❌ OpenAI embeddings error:", e)
            return jsonify({"error": "embedding_error"}), 500

        # 2) Query Pinecone for relevant chunks
        try:
            results = index.query(vector=q_embed, top_k=TOP_K, include_metadata=True)
        except Exception as e:
            print("❌ Pinecone query error:", e)
            return jsonify({"error": "pinecone_error"}), 500

        retrieved_texts = []
        sources = []
        matches = getattr(results, "matches", []) or results.matches if hasattr(results, "matches") else []
        # support different Pinecone return shapes
        for m in matches:
            metadata = getattr(m, "metadata", {}) or m.metadata
            text = metadata.get("text") or metadata.get("content") or ""
            url = metadata.get("url") or metadata.get("page") or ""
            if text:
                # include small label with source
                prefix = f"Source: {url}\n" if url else ""
                retrieved_texts.append(f"{prefix}{text}")
                if url:
                    sources.append(url)

        context_block = "\n\n---\n\n".join(retrieved_texts) if retrieved_texts else ""

        # 3) Conversation history (last N messages)
        recent_msgs = get_recent_messages(session_id)

        # Build messages for the chat completion
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context_block:
            messages.append({"role": "system", "content": f"Website content (only this may be used):\n\n{context_block}"})

        for m in recent_msgs:
            # include previous 'user' and 'assistant' messages
            messages.append({"role": m["role"], "content": m["content"]})

        # current user message
        messages.append({"role": "user", "content": question})

        # 4) Ask OpenAI (chat completion)
        try:
            completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            print("❌ OpenAI chat completion error:", e)
            return jsonify({"error": "openai_error"}), 500

        # persist assistant response
        save_message(session_id, "assistant", answer)

        # Return answer and unique sources list
        unique_sources = list(dict.fromkeys([s for s in sources if s]))
        return jsonify({"answer": answer, "sources": unique_sources})
    except Exception as e:
        print("❌ Unexpected error in /chat:", e)
        return jsonify({"error": "server_error"}), 500

# Save chat (alternate route to save full transcript from frontend)
@app.route("/save-chat", methods=["POST"])
def save_chat():
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id")
        name = data.get("name")
        phone = data.get("phone")
        chat_text = data.get("chat")
        page = data.get("page")

        if not session_id or not name or not phone or not chat_text:
            return jsonify({"error": "Missing required fields"}), 400

        # ensure session exists and update
        s = Session.query.filter_by(session_id=session_id).first()
        if not s:
            s = Session(session_id=session_id, name=name, phone=phone, page=page)
            db.session.add(s)
        else:
            s.name = name
            s.phone = phone
            if page:
                s.page = page
        db.session.commit()

        # Optionally save the chat_text as a single Message row (role 'system' or custom)
        save_message(session_id, "system", chat_text)

        return jsonify({"message": "Chat saved successfully!"})
    except Exception as e:
        print("❌ Error in /save-chat:", e)
        return jsonify({"error": "server_error"}), 500

# Admin dashboard (basic auth)
@app.route("/admin/sessions", methods=["GET"])
@require_basic_auth
def admin_sessions():
    try:
        sessions = Session.query.order_by(Session.created_at.desc()).limit(200).all()
        out = []
        for s in sessions:
            last_msg = Message.query.filter_by(session_id=s.session_id).order_by(Message.id.desc()).first()
            out.append({
                "session_id": s.session_id,
                "name": s.name,
                "phone": s.phone,
                "page": s.page,
                "created_at": s.created_at.isoformat(),
                "last_message": last_msg.content if last_msg else None
            })
        return jsonify(out)
    except Exception as e:
        print("❌ Error in admin_sessions:", e)
        return jsonify({"error": "server_error"}), 500

@app.route("/admin/session/<session_id>", methods=["GET"])
@require_basic_auth
def admin_view_session(session_id):
    try:
        s = Session.query.filter_by(session_id=session_id).first_or_404()
        msgs = Message.query.filter_by(session_id=session_id).order_by(Message.id.asc()).all()
        return jsonify({
            "session": {
                "session_id": s.session_id,
                "name": s.name,
                "phone": s.phone,
                "page": s.page,
                "created_at": s.created_at.isoformat()
            },
            "messages": [{"role": m.role, "content": m.content, "time": m.created_at.isoformat()} for m in msgs]
        })
    except Exception as e:
        print("❌ Error in admin_view_session:", e)
        return jsonify({"error": "server_error"}), 500

# health
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # On Render, use gunicorn in start command; this is only used for local testing
    app.run(host="0.0.0.0", port=PORT)
