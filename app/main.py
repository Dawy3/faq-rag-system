import logging
from fastapi import FastAPI
from app.api.v1.endpoints import router as api_router

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Production FAQ RAG System")

#Include Routers
app.include_router(api_router, prefix="/api/v1")


    