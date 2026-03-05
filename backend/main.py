from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine
from . import models
from .routes.auth_routes import router as auth_router
from .routes.sip_routes import router as sip_router

# Create all database tables on startup
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Water Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "app://.", "file://"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(sip_router)


@app.get("/health")
def health():
    return {"status": "ok"}
