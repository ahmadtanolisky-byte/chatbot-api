# models.py
import datetime
from config import db

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
