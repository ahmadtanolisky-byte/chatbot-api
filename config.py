# config.py
import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------- Configuration ----------
PORT = int(os.getenv("PORT", 5000))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chats.db")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")
TOP_K = int(os.getenv("TOP_K", 8))
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", 6))
RATE_LIMIT = os.getenv("RATE_LIMIT", "60 per minute")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 3500))
SHORT_QUERY_WORDS = int(os.getenv("SHORT_QUERY_WORDS", 4))

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

# ---------- Flask App ----------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

db = SQLAlchemy(app)
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
limiter.init_app(app)

# ---------- OpenAI ----------
client = OpenAI(api_key=OPENAI_API_KEY)
print("✅ OpenAI client initialized")

# ---------- Pinecone ----------
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
print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' initialized")

