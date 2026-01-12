import os
import datetime
import logging
from functools import wraps
from typing import List

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from flask import render_template

from openai import OpenAI

import pinecone

load_dotenv()

PORT = int(os.getenv("PORT", 5000))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") 
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL not set! Please add it in Render Environment.")

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")
TOP_K = int(os.getenv("TOP_K", 7)) 
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", 6))
RATE_LIMIT = os.getenv("RATE_LIMIT", "60 per minute")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 4500))  
SHORT_QUERY_WORDS = int(os.getenv("SHORT_QUERY_WORDS", 4))  

if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in environment")
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise RuntimeError("Please set PINECONE_API_KEY and PINECONE_INDEX_NAME in environment")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGIN", "").split(",") if o.strip()]
if not allowed_origins:
    allowed_origins = ["*"]

CORS(
    app,
    origins=allowed_origins,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return ("", 200)

db = SQLAlchemy(app)

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
limiter.init_app(app)

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI client initialized")
except Exception as e:
    print("❌ OpenAI init error:", e)
    raise

try:
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"⚠️ Index '{PINECONE_INDEX_NAME}' not found. Creating it automatically...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  
        )

    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' initialized successfully")

except Exception as e:
    print("❌ Pinecone init error:", e)
    raise


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
    role = db.Column(db.String(32))  
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

with app.app_context():
    db.create_all()

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
        if content and len(content) > 20000:
            content = content[:20000] + "...(truncated)"
        m = Message(session_id=session_id, role=role, content=content)
        db.session.add(m)
        db.session.commit()
    except Exception as e:
        logger.exception("❌ Failed to save message:")

def get_recent_messages(session_id: str, limit: int = CONTEXT_MESSAGES) -> List[dict]:
    msgs = Message.query.filter_by(session_id=session_id).order_by(Message.id.desc()).limit(limit).all()
    return list(reversed([{"role": m.role, "content": m.content} for m in msgs]))

def trim_context_text(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]

SYSTEM_PROMPT = (
    "You are a helpful assistant for the Sky Marketing real estate website. "
    "Answer using ONLY the provided website content and conversation context. "
    "If the user refers to a block, phase, or society, use the context to determine which one. "
    "Do NOT assume anything not in context. "
    "Also read user previous chat and reply according to it "
    "All details are uploaded in pinecone in form of chunks, almost 1200 chunks, Get data from there and reply according to it "
    "If the answer is not in the context, reply: "
    "'I'm sorry, I don't have that exact information yet. Would you like me to connect you with our sales team?' "
    "Keep answers short, factual, and reference source URLs when appropriate."
)

GENERIC_SHORT_QUERIES = {"payment plan", "map", "location", "plots", "details", "price", "where", "on cash", "cash"}

def is_short_query(q: str) -> bool:
    return len(q.split()) < SHORT_QUERY_WORDS or q.lower() in GENERIC_SHORT_QUERIES

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Chatbot API running"})

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})

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
        logger.exception("❌ /session error:")
        return jsonify({"error": "server_error"}), 500

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

        question = " ".join(question.split())

        save_message(session_id, "user", question)

        q_lower = question.lower()
        if q_lower in {"hi", "hello", "hey", "salam", "assalamualaikum"}:
            s = Session.query.filter_by(session_id=session_id).first()
            name = s.name if s else ""
            answer = f"👋 Hello {name}! How can I help you today? Ask me about plots, payment plans, or booking info."
            save_message(session_id, "assistant", answer)
            return jsonify({"answer": answer})

        last_msgs = get_recent_messages(session_id, limit=CONTEXT_MESSAGES)
        last_user_msgs = [m["content"] for m in last_msgs if m["role"] == "user"]

        recent_text = " ".join(m["content"].lower() for m in last_msgs)
        KNOWN_BLOCKS = ["n block", "a block", "b block", "faisal town", "phase 2", "phase ii", "phase 1", "central business district", "cbd"]



        try:
            context_text = " ".join(last_user_msgs[-3:]) 
            context_text = trim_context_text(context_text, max_chars=MAX_CONTEXT_CHARS)

            if is_short_query(question):
                embedding_input = (context_text + " " + question).strip()
            else:
                embedding_input = question

            if not embedding_input:
                embedding_input = question

            embed_resp = client.embeddings.create(model="text-embedding-3-small", input=embedding_input)
            q_embed = embed_resp.data[0].embedding
        except Exception as e:
            logger.exception("❌ OpenAI embedding error:")
            return jsonify({"error": "embedding_error"}), 500

        try:
            results = index.query(vector=q_embed, top_k=TOP_K, include_metadata=True)
        except Exception as e:
            logger.exception("❌ Pinecone query error:")
            return jsonify({"error": "pinecone_error"}), 500

        retrieved_texts = []
        sources = []
        matches = []
        if hasattr(results, "matches"):
            matches = results.matches or []
        elif isinstance(results, dict) and "matches" in results:
            matches = results["matches"] or []

        for m in matches:
            metadata = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else {}) or {}
            text = metadata.get("text") or metadata.get("content") or metadata.get("page_text") or ""
            url = metadata.get("url") or metadata.get("page") or metadata.get("source") or ""
            if not text:
                text = getattr(m, "value", "") or (m.get("payload") if isinstance(m, dict) else "") or ""
            if text:
                prefix = f"Source: {url}\n" if url else ""
                retrieved_texts.append(f"{prefix}{text}")
                if url:
                    sources.append(url)

        context_block = "\n\n---\n\n".join(retrieved_texts) if retrieved_texts else ""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context_block:
            messages.append({"role": "system", "content": f"Website content (only use this to answer):\n\n{trim_context_text(context_block, max_chars=MAX_CONTEXT_CHARS)}"})

        for m in last_msgs:
            role = m["role"]
            if role not in {"user", "assistant", "system"}:
                role = "user"
            messages.append({"role": role, "content": m["content"]})

        messages.append({"role": "user", "content": question})

        if len(messages) > (CONTEXT_MESSAGES + 5):
            system_msgs = [m for m in messages if m["role"] == "system"]
            other_msgs = [m for m in messages if m["role"] != "system"]
            messages = system_msgs + other_msgs[-(CONTEXT_MESSAGES + 3):]

        try:
            completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2, max_tokens=800)
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("❌ OpenAI completion error:")
            return jsonify({"error": "openai_error"}), 500
        

        save_message(session_id, "assistant", answer)

        unique_sources = []
        seen = set()
        for s in sources:
            if s and s not in seen:
                unique_sources.append(s)
                seen.add(s)

        return jsonify({"answer": answer, "sources": unique_sources})
    except Exception as e:
        logger.exception("❌ Unexpected /chat error:")
        return jsonify({"error": "server_error"}), 500

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

        save_message(session_id, "system", chat_text)

        return jsonify({"message": "Chat saved successfully!"})
    except Exception as e:
        logger.exception("❌ /save-chat error:")
        return jsonify({"error": "server_error"}), 500

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
        logger.exception("❌ admin_sessions error:")
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
        logger.exception("❌ admin_view_session error:")
        return jsonify({"error": "server_error"}), 500
    

from flask import render_template

@app.route("/admin/dashboard")
@require_basic_auth
def admin_dashboard():
    return render_template("dashboard.html")
@app.route("/db-info")
def db_info():
    return {"database_url": app.config["SQLALCHEMY_DATABASE_URI"]}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
