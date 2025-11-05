from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
CORS(app)

# Load keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

@app.route("/")
def home():
    return "✅ Chatbot API is running successfully!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Step 1: Create embedding for question
    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # Step 2: Search Pinecone
    results = index.query(vector=q_embed, top_k=3, include_metadata=True)

    # Step 3: Create context from website data
    context = "\n".join([m.metadata["text"] for m in results.matches])

    # Step 4: Ask GPT
    prompt = f"""
    You are a helpful assistant that answers using only this website data:
    {context}

    Question: {question}
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = completion.choices[0].message.content.strip()
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
