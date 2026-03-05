from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sip_events = relationship("SipEvent", back_populates="user")


class SipEvent(Base):
    __tablename__ = "sip_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(Float, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    volume_ml = Column(Float, nullable=False)
    sip_count = Column(Integer, nullable=False)

    user = relationship("User", back_populates="sip_events")
