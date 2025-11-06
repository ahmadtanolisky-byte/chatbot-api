from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os

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

# Temporary storage (can later be replaced with database)
chats = []

@app.route("/")
def home():
    return "✅ Chatbot API is running successfully!"

# ==== MAIN CHAT ROUTE ====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 1️⃣ Create embedding for the question
    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # 2️⃣ Search Pinecone
    results = index.query(vector=q_embed, top_k=3, include_metadata=True)

    # 3️⃣ Build context from website data
    context = "\n".join([m.metadata["text"] for m in results.matches])

    # 4️⃣ Ask GPT
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


# ==== SAVE CHAT (USER DETAILS + CHAT LOG) ====
@app.route("/save-chat", methods=["POST"])
def save_chat():
    data = request.get_json()
    name = data.get("name")
    phone = data.get("phone")
    chat = data.get("chat")
    page = data.get("page")

    if not all([name, phone, chat, page]):
        return jsonify({"error": "Missing required fields"}), 400

    chats.append({"name": name, "phone": phone, "chat": chat, "page": page})
    return jsonify({"message": "Chat saved successfully!"})


# ==== DASHBOARD TO VIEW CHATS z====
@app.route("/dashboard", methods=["GET"])
def dashboard():
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
    for c in chats:
        html += f"""
        <tr>
            <td>{c['name']}</td>
            <td>{c['phone']}</td>
            <td><a href='{c['page']}' target='_blank'>Open Page</a></td>
            <td style='white-space: pre-wrap;'>{c['chat']}</td>
        </tr>
        """
    html += "</table>"
    return html
  

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
