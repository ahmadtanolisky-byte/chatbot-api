# routes.py
from flask import jsonify, request
from config import app, limiter, client, index, logger, TOP_K, CONTEXT_MESSAGES, MAX_CONTEXT_CHARS
from models import Session, Message
from utils import save_message, get_recent_messages, trim_context_text, require_basic_auth

@app.route("/chat", methods=["POST"])
@limiter.limit("60 per minute")
def chat():
    try:
        data = request.get_json(force=True) or {}
        session_id = data.get("session_id")
        question = (data.get("question") or "").strip()

        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        if not question:
            return jsonify({"error": "question required"}), 400

        # Normalize whitespace
        question = " ".join(question.split())

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

        # ---------- Context-aware enrichment ----------
        # get last few messages and last few user messages
        last_msgs = get_recent_messages(session_id, limit=CONTEXT_MESSAGES)
        last_user_msgs = [m["content"] for m in last_msgs if m["role"] == "user"]

        # try to detect previously mentioned block/area automatically for short/generic queries
        recent_text = " ".join(m["content"].lower() for m in last_msgs)
        # small list - you can extend with more blocks / project names
        KNOWN_BLOCKS = ["n block", "a block", "b block", "faisal town", "phase 2", "phase ii", "phase 1", "central business district", "cbd"]

        if question.lower() in GENERIC_SHORT_QUERIES or is_short_query(question):
            for block in KNOWN_BLOCKS:
                if block in recent_text:
                    # append the most-likely block to the question (so retrieval improves)
                    question = f"{question} {block}"
                    logger.debug("Enriched short question with context: %s", question)
                    break

        # ---------- 1) Create context-aware embedding ----------
        try:
            # Build a short context string from recent user messages (most recent ones matter more)
            context_text = " ".join(last_user_msgs[-3:])  # use last 3 user messages
            context_text = trim_context_text(context_text, max_chars=MAX_CONTEXT_CHARS)

            # If the question is short or ambiguous, combine with recent user context for embedding search
            if is_short_query(question):
                embedding_input = (context_text + " " + question).strip()
            else:
                embedding_input = question

            # safety fallback
            if not embedding_input:
                embedding_input = question

            embed_resp = client.embeddings.create(model="text-embedding-3-small", input=embedding_input)
            q_embed = embed_resp.data[0].embedding
        except Exception as e:
            logger.exception("❌ OpenAI embedding error:")
            return jsonify({"error": "embedding_error"}), 500

        # ---------- 2) Pinecone query ----------
        try:
            results = index.query(vector=q_embed, top_k=TOP_K, include_metadata=True)
        except Exception as e:
            logger.exception("❌ Pinecone query error:")
            return jsonify({"error": "pinecone_error"}), 500

        # ---------- 3) collect retrieved context + sources ----------
        retrieved_texts = []
        sources = []
        matches = []
        # results may be dict-like or object depending on client version
        if hasattr(results, "matches"):
            matches = results.matches or []
        elif isinstance(results, dict) and "matches" in results:
            matches = results["matches"] or []

        for m in matches:
            # metadata can be nested differently in various Pinecone setups; handle multiple shapes
            metadata = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else {}) or {}
            text = metadata.get("text") or metadata.get("content") or metadata.get("page_text") or ""
            url = metadata.get("url") or metadata.get("page") or metadata.get("source") or ""
            if not text:
                # sometimes the match object itself has a 'value' or 'payload'
                text = getattr(m, "value", "") or (m.get("payload") if isinstance(m, dict) else "") or ""
            if text:
                prefix = f"Source: {url}\n" if url else ""
                retrieved_texts.append(f"{prefix}{text}")
                if url:
                    sources.append(url)

        context_block = "\n\n---\n\n".join(retrieved_texts) if retrieved_texts else ""

        # ---------- 4) build messages including recent conversation for context ----------
        # include the system prompt + website context (system role) + recent chat messages + current user message
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context_block:
            # Keep the context block trimmed to avoid hitting token limits
            messages.append({"role": "system", "content": f"Website content (only use this to answer):\n\n{trim_context_text(context_block, max_chars=MAX_CONTEXT_CHARS)}"})

        # append the recent chat messages (already in chronological order)
        # ensure roles are valid (user/assistant/system)
        for m in last_msgs:
            role = m["role"]
            if role not in {"user", "assistant", "system"}:
                role = "user"
            messages.append({"role": role, "content": m["content"]})

        # the current user question is the last message
        messages.append({"role": "user", "content": question})

        # Optional: If the messages are very long, keep only the last N items
        if len(messages) > (CONTEXT_MESSAGES + 5):
            # preserve system prompts at front and keep last CONTEXT_MESSAGES messages
            system_msgs = [m for m in messages if m["role"] == "system"]
            other_msgs = [m for m in messages if m["role"] != "system"]
            messages = system_msgs + other_msgs[-(CONTEXT_MESSAGES + 3):]

        # ---------- 5) Ask OpenAI chat completions ----------
        try:
            # Use a deterministic-ish temperature for factual answers
            completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2, max_tokens=800)
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("❌ OpenAI completion error:")
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
        logger.exception("❌ Unexpected /chat error:")
        return jsonify({"error": "server_error"}), 500
