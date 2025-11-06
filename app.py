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
    return "✅ Chatbot API is running successfully with session-based chat storage!"

# ==== CHAT ROUTE ====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # === Embed + Query Pinecone ===
    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    results = index.query(vector=q_embed, top_k=3, include_metadata=True)
    context = "\n".join([m.metadata["text"] for m in results.matches])

    prompt = f"""
    You are a helpful assistant that answers using the following website data:
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

    # ✅ If chat already exists (same name & phone), append messages instead of creating new
    cursor.execute("SELECT chat FROM chats WHERE name=? AND phone=?", (name, phone))
    existing = cursor.fetchone()

    if existing:
        updated_chat = existing[0] + "\n" + chat
        cursor.execute("UPDATE chats SET chat=? WHERE name=? AND phone=?", (updated_chat, name, phone))
    else:
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
    cursor.execute("SELECT id, name, phone, page, chat FROM chats ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    html = """
    <h2>💬 Chatbot Dashboard</h2>
    <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%;'>
    <tr style='background:#0073e6;color:white;'>
        <th>ID</th>
        <th>Name</th>
        <th>Phone</th>
        <th>Page Link</th>
        <th>View</th>
    </tr>
    """
    for id, name, phone, page, chat in rows:
        html += f"""
        <tr>
            <td>{id}</td>
            <td>{name}</td>
            <td>{phone}</td>
            <td><a href='{page}' target='_blank'>Open Page</a></td>
            <td><a href='/chat/{id}' target='_blank'>👁️ View Chat</a></td>
        </tr>
        """
    html += "</table>"
    return html


# ==== SINGLE CHAT VIEW ====
@app.route("/chat/<int:chat_id>", methods=["GET"])
def view_chat(chat_id):
    conn = sqlite3.connect("chats.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, phone, page, chat FROM chats WHERE id=?", (chat_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return "<h3>❌ Chat not found.</h3>"

    name, phone, page, chat = row

    html = f"""
    <h2>💬 Chat Details</h2>
    <p><b>Name:</b> {name}</p>
    <p><b>Phone:</b> {phone}</p>
    <p><b>Page:</b> <a href='{page}' target='_blank'>{page}</a></p>
    <hr>
    <pre style='background:#f5f5f5;padding:15px;border-radius:8px;white-space:pre-wrap;'>{chat}</pre>
    <p><a href='/dashboard'>⬅️ Back to Dashboard</a></p>
    """
    return html


# ==== RUN ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
