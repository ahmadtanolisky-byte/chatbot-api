# app.py - production-ready chatbot backend
import os
import datetime
from functools import wraps

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ---------- Configuration (from env) ----------
PORT = int(os.getenv("PORT", 5000))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chats.db")  # replace with Postgres URL in production
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")
TOP_K = int(os.getenv("TOP_K", 8))  # increased recall
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", 6))
RATE_LIMIT = os.getenv("RATE_LIMIT", "60 per minute")

# Validate required env vars
if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise RuntimeError("Please set OPENAI_API_KEY, PINECONE_API_KEY and PINECONE_INDEX_NAME in environment")

# ---------- Flask + Extensions ----------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

db = SQLAlchemy(app)

# Flask-Limiter v3 style
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
limiter.init_app(app)

# ---------- OpenAI & Pinecone clients ----------
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print("❌ OpenAI init error:", e)
    raise

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print("❌ Pinecone init error:", e)
    raise

# ---------- Database models ----------
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
    role = db.Column(db.String(32))  # 'user', 'assistant', 'system'
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# create tables
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

def save_message(session_id: str, role: str, content: str):
    try:
        m = Message(session_id=session_id, role=role, content=content)
        db.session.add(m)
        db.session.commit()
    except Exception as e:
        # do not crash on DB write errors; log and continue
        print("❌ Failed to save message:", e)

def get_recent_messages(session_id: str, limit: int = CONTEXT_MESSAGES):
    msgs = Message.query.filter_by(session_id=session_id).order_by(Message.id.desc()).limit(limit).all()
    return list(reversed([{"role": m.role, "content": m.content} for m in msgs]))

# Strict but practical system prompt (tune as needed)
SYSTEM_PROMPT = (
    "You are a helpful assistant for the Faisal Town / Sky Marketing website. "
    "Answer using only the provided website content. If you cannot find a direct answer, reply: "
    "'I'm sorry, I don't have that exact information yet. Would you like me to connect you with our sales team?' "
    "Keep answers short, factual, and reference source URLs when appropriate."
)

# ---------- Routes ----------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Chatbot API running"})

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})

# start/update session
@app.route("/session", methods=["POST"])
@limiter.limit("10 per minute")
def start_session():
    try:
        data = request.get_json(force=True) or {}
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
            if name: s.name = name
            if phone: s.phone = phone
            if page: s.page = page

        db.session.commit()
        return jsonify({"message": "session started", "session_id": session_id})
    except Exception as e:
        print("❌ /session error:", e)
        return jsonify({"error": "server_error"}), 500

# main chat endpoint
@app.route("/chat", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def chat():
    try:
        data = request.get_json(force=True) or {}
        session_id = data.get("session_id")
        question = (data.get("question") or "").strip()

        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        if not question:
            return jsonify({"error": "question required"}), 400

        # save user message
        save_message(session_id, "user", question)

        # quick greetings/fallback handling
        q_lower = question.lower()
        if q_lower in {"hi", "hello", "hey", "salam", "assalamualaikum"}:
            s = Session.query.filter_by(session_id=session_id).first()
            name = s.name if s else ""
            answer = f"👋 Hello {name}! How can I help you today? Ask me about plots, payment plans, or booking info."
            save_message(session_id, "assistant", answer)
            return jsonify({"answer": answer})
        
        last_msgs = get_recent_messages(session_id, limit=3)
        recent_text = " ".join(m["content"].lower() for m in last_msgs)

        # If user asks generic things like "payment plan" or "map"
        if question.lower() in ["payment plan", "map", "location", "plots", "details"]:
            # Try to find a block name in the recent context
            for block in ["n block", "a block", "b block", "faisal town", "phase 2"]:
                if block in recent_text:
                    question = f"{question} {block}"
                    break


        # 1) create embedding
        try:
            # Combine recent user context with the current question
            recent_msgs = get_recent_messages(session_id, limit=CONTEXT_MESSAGES)
            context_text = " ".join([
                m["content"] for m in recent_msgs
                if m["role"] == "user"
            ][-3:])  # last 3 user messages

            # If the user is asking something short, enrich it
            if len(question.split()) < 4:  # e.g. "location", "on cash"
                context_query = (context_text + " " + question).strip()
            else:
                context_query = question

            # Create embedding based on context-aware query
            embed_resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=context_query
            )
            q_embed = embed_resp.data[0].embedding

        except Exception as e:
            print("❌ OpenAI embedding error:", e)
            return jsonify({"error": "embedding_error"}), 500

        # 2) pinecone query
        try:
            results = index.query(vector=q_embed, top_k=TOP_K, include_metadata=True)
        except Exception as e:
            print("❌ Pinecone query error:", e)
            return jsonify({"error": "pinecone_error"}), 500

        # 3) collect retrieved context + sources
        retrieved_texts = []
        sources = []
        # handle different possible shapes of results
        matches = []
        if hasattr(results, "matches"):
            matches = results.matches or []
        elif isinstance(results, dict) and "matches" in results:
            matches = results["matches"] or []

        for m in matches:
            metadata = getattr(m, "metadata", {}) or (m.get("metadata") if isinstance(m, dict) else {})
            text = metadata.get("text") or metadata.get("content") or ""
            url = metadata.get("url") or metadata.get("page") or ""
            if text:
                prefix = f"Source: {url}\n" if url else ""
                retrieved_texts.append(f"{prefix}{text}")
                if url:
                    sources.append(url)

        context_block = "\n\n---\n\n".join(retrieved_texts) if retrieved_texts else ""

        # 4) build messages including recent conversation for context
        recent_messages = get_recent_messages(session_id, limit=CONTEXT_MESSAGES)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context_block:
            messages.append({"role": "system", "content": f"The following website data may help:\n\n{context_block}"})
        messages.extend(recent_messages)
        messages.append({"role": "user", "content": question})



        # 5) ask OpenAI
        try:
            completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            print("❌ OpenAI completion error:", e)
            return jsonify({"error": "openai_error"}), 500

        # persist assistant reply
        save_message(session_id, "assistant", answer)

        # dedupe sources preserving order
        unique_sources = []
        seen = set()
        for s in sources:
            if s and s not in seen:
                unique_sources.append(s)
                seen.add(s)

        return jsonify({"answer": answer, "sources": unique_sources})

    except Exception as e:
        print("❌ Unexpected /chat error:", e)
        return jsonify({"error": "server_error"}), 500

# save chat (full transcript save)
@app.route("/save-chat", methods=["POST"])
def save_chat():
    try:
        data = request.get_json(force=True) or {}
        session_id = data.get("session_id")
        name = data.get("name")
        phone = data.get("phone")
        chat_text = data.get("chat")
        page = data.get("page")

        if not session_id or not name or not phone or not chat_text:
            return jsonify({"error": "Missing required fields"}), 400

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

        # Optionally record a 'system' message that contains full transcript snapshot
        save_message(session_id, "system", chat_text)

        return jsonify({"message": "Chat saved successfully!"})
    except Exception as e:
        print("❌ /save-chat error:", e)
        return jsonify({"error": "server_error"}), 500

# Admin endpoints
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
        print("❌ admin_sessions error:", e)
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
        print("❌ admin_view_session error:", e)
        return jsonify({"error": "server_error"}), 500

# ---------- Run ----------
if __name__ == "__main__":
    # For local testing only; in production use gunicorn: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
    app.run(host="0.0.0.0", port=PORT)
