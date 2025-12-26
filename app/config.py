import os 
from dotenv import load_dotenv
load_dotenv()

class AppConfig:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
    INDEX_NAME = "faq-rag-prod"
    EMBEDDING_MDOEL = "sentence-transformers/all-MiniLM-L6-v2"
    
config = AppConfig()


    