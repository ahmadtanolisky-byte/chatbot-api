# config.py - configuration and setup

import os
import logging
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==== APP & LOGGING ====
from app import app
logger = logging.getLogger(__name__)

# ==== DATABASE ====
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///chatbot.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ==== RATE LIMITER ====
limiter = Limiter(get_remote_address, app=app, default_limits=["20/minute"])

# ==== KEYS ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "skymarketing")

# ==== CLIENTS ====
logger.info("✅ OpenAI client initialized")
client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)
logger.info(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' initialized")

# ==== CONSTANTS ====
TOP_K = 10
CONTEXT_MESSAGES = 6
MAX_CONTEXT_CHARS = 8000
SHORT_QUERY_WORDS = 4

# ==== PROMPT & HELPERS ====
SYSTEM_PROMPT = (
    "You are a helpful assistant for the Faisal Town / Sky Marketing website. "
    "Answer using only the provided website content and conversation context. "
    "If you cannot find a direct answer, reply: "
    "'I'm sorry, I don't have that exact information yet. Would you like me to connect you with our sales team?' "
    "Keep answers short, factual, and reference source URLs when appropriate."
)

GENERIC_SHORT_QUERIES = {"payment plan", "map", "location", "plots", "details", "price", "where", "on cash", "cash"}

def is_short_query(q: str, SHORT_QUERY_WORDS=SHORT_QUERY_WORDS) -> bool:
    if not q:
        return False
    return len(q.split()) < SHORT_QUERY_WORDS or q.lower() in GENERIC_SHORT_QUERIES
