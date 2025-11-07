# utils.py
from functools import wraps
from flask import request, Response
from config import ADMIN_USER, ADMIN_PASS, db, logger
from models import Message

def require_basic_auth(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != ADMIN_USER or auth.password != ADMIN_PASS:
            return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})
        return func(*args, **kwargs)
    return wrapped

def save_message(session_id, role, content):
    try:
        if content and len(content) > 20000:
            content = content[:20000] + "...(truncated)"
        msg = Message(session_id=session_id, role=role, content=content)
        db.session.add(msg)
        db.session.commit()
    except Exception as e:
        logger.exception("❌ Failed to save message:")

def get_recent_messages(session_id, limit=6):
    msgs = Message.query.filter_by(session_id=session_id).order_by(Message.id.desc()).limit(limit).all()
    return list(reversed([{"role": m.role, "content": m.content} for m in msgs]))

def trim_context_text(text, max_chars=3500):
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]
