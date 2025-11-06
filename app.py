from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import sqlite3

load_dotenv()
app = Flask(__name__)
CORS(app)

# ==== KEYS ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ==== CLIENTS ====
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ==== DATABASE SETUP ====
def init_db():
    conn = sqlite3.connect("chats.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            phone TEXT,
            page TEXT,
            chat TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==== HOME ====
@app.route("/")
def home():
    return "✅ Chatbot API is running successfully with SQLite storage!"

# ==== CHAT ROUTE ====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    results = index.query(vector=q_embed, top_k=3, include_metadata=True)
    context = "\n".join([m.metadata["text"] for m in results.matches])

    prompt = f"""
    You are a helpful assistant that answers only using the following website data:
    {context}

    Question: {question}
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = completion.choices[0].message.content.strip()
    return jsonify({"answer": answer})

# ==== SAVE CHAT ====
@app.route("/save-chat", methods=["POST"])
def save_chat():
    data = request.get_json()
    name = data.get("name")
    phone = data.get("phone")
    chat = data.get("chat")
    page = data.get("page")

    if not all([name, phone, chat, page]):
        return jsonify({"error": "Missing required fields"}), 400

    conn = sqlite3.connect("chats.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (name, phone, page, chat) VALUES (?, ?, ?, ?)",
                   (name, phone, page, chat))
    conn.commit()
    conn.close()

    return jsonify({"message": "Chat saved successfully!"})

# ==== DASHBOARD ====
@app.route("/dashboard", methods=["GET"])
def dashboard():
    conn = sqlite3.connect("chats.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, phone, page, chat FROM chats ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    html = """
    <h2>💬 Chatbot Dashboard</h2>
    <table border='1' cellpadding='8' cellspacing='0'>
    <tr style='background:#0073e6;color:white;'>
        <th>Name</th>
        <th>Phone</th>
        <th>Page Link</th>
        <th>Chat</th>
    </tr>
    """
    for name, phone, page, chat in rows:
        html += f"""
        <tr>
            <td>{name}</td>
            <td>{phone}</td>
            <td><a href='{page}' target='_blank'>Open Page</a></td>
            <td style='white-space: pre-wrap;'>{chat}</td>
        </tr>
        """
    html += "</table>"
    return html

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
