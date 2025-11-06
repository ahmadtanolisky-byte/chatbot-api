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
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chats.db")  # Use Postgres URL in production
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")  # set to https://your-website.com in prod
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")
TOP_K = int(os.getenv("TOP_K", 6))  # number of Pinecone chunks to retrieve
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", 6))  # last N messages included for context
RATE_LIMIT = os.getenv("RATE_LIMIT", "20 per minute")  # tune as needed

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise RuntimeError("OPENAI_API_KEY, PINECONE_API_KEY and PINECONE_INDEX_NAME must be set as env vars")

# ---------- Flask + Extensions ----------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

db = SQLAlchemy(app)
limiter = Limiter(app, key_func=get_remote_address, default_limits=[RATE_LIMIT])

# ---------- OpenAI & Pinecone Clients ----------
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

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

# create tables if not exist
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
    m = Message(session_id=session_id, role=role, content=content)
    db.session.add(m)
    db.session.commit()

# Strict assistant prompt (will be combined with retrieved context + conversation history)
SYSTEM_PROMPT = (
    "You are a helpful assistant that must ONLY answer using the provided website content. "
    "If the answer cannot be found in the provided content, respond exactly: 'I don't know.' "
    "Be concise and return practical, factual replies. If you include a link, use the original page URLs provided."
)

# ---------- API Routes ----------

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Chatbot API is running."})

# Start or update session (create session record if not exists)
@app.route("/session", methods=["POST"])
@limiter.limit("10 per minute")
def start_session():
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

# Main chat endpoint: takes session_id and question, returns answer and stores messages
@app.route("/chat", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def chat():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    question = data.get("question", "")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    if not question:
        return jsonify({"error": "question required"}), 400

    # store user message
    save_message(session_id, "user", question)

    # 1) Get embedding for the question
    q_embed = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding

    # 2) Query Pinecone for relevant chunks
    results = index.query(vector=q_embed, top_k=TOP_K, include_metadata=True)
    retrieved_texts = []
    sources = []
    for m in results.matches:
        text = m.metadata.get("text") or m.metadata.get("content") or ""
        url = m.metadata.get("url") or m.metadata.get("page") or ""
        if text:
            # include small label with source
            retrieved_texts.append(f"Source: {url}\n{text}")
            if url:
                sources.append(url)

    context_block = "\n\n---\n\n".join(retrieved_texts) if retrieved_texts else ""

    # 3) Conversation history (last N messages)
    recent_msgs = get_recent_messages(session_id)
    # convert to the chat.completions message structure
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add fetched context as a system/assistant note so model uses it
    if context_block:
        messages.append({"role": "system", "content": f"Website content (only this may be used):\n\n{context_block}"})

    # append recent conversation history
    for m in recent_msgs:
        messages.append({"role": m["role"], "content": m["content"]})

    # append current user message (again) for clarity
    messages.append({"role": "user", "content": question})

    # 4) Ask OpenAI (chat completion)
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    answer = completion.choices[0].message.content.strip()

    # If model answered with more than allowed, trim as needed (optional)
    save_message(session_id, "assistant", answer)

    # Return answer and sources list (URLs)
    return jsonify({"answer": answer, "sources": list(dict.fromkeys(sources))})

# Admin dashboard (basic auth)
@app.route("/admin/sessions", methods=["GET"])
@require_basic_auth
def admin_sessions():
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

@app.route("/admin/session/<session_id>", methods=["GET"])
@require_basic_auth
def admin_view_session(session_id):
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

# health
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
