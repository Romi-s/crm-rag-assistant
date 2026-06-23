"""Central configuration (pydantic-settings).

All values can be overridden via environment variables or a local `.env` file.
The two things worth understanding here:

1. `llm_provider` selects the *generation* backend ("local" Ollama now, "bedrock"
   in Part 2). It does NOT affect embeddings.
2. Embeddings are ALWAYS local (fastembed). This keeps the vector index
   provider-agnostic: you never re-embed the corpus when switching local<->bedrock,
   and embeddings stay free. Only answer generation changes between modes.
"""

from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Export .env into the real process environment. pydantic-settings already reads
# .env for our OWN fields, but boto3 (Bedrock) reads AWS_* straight from os.environ,
# so this is what makes AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env reach AWS.
# Does not override variables already set in the real environment.
load_dotenv()


class Settings(BaseSettings):
    # --- Generation provider ---------------------------------------------------
    # "local"  -> Ollama (open-source model on this machine)
    # "bedrock"-> Amazon Bedrock (implemented in Part 2)
    llm_provider: str = "local"

    # Local / Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_timeout: int = 180          # CPU inference can be slow; be generous
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.1   # low -> stay grounded, less invention

    # Bedrock (Part 2). Credentials come from the standard AWS chain (env / ~/.aws),
    # never from this file. us-east-1 has the broadest Bedrock model availability.
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"

    # --- Embeddings (always local) --------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"   # fastembed ONNX, CPU-friendly
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # --- Vector store (ChromaDB) ----------------------------------------------
    chroma_host: Optional[str] = None
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_token: Optional[str] = None
    chroma_persist_dir: str = "./chroma_data"
    collection_name: str = "crm_docs"

    # --- Chunking & retrieval --------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_pdf_pages: int = 100

    retrieval_top_k: int = 12     # candidates pulled per method (vector / BM25)
    final_top_k: int = 6          # chunks actually sent to the LLM
    entity_top_k: int = 8         # max records force-included for a named customer

    # --- Dataset ---------------------------------------------------------------
    # Folder that holds the provided dataset (crm_records/ sales/ tickets/ ...).
    dataset_dir: str = "./crm_rag_assistance"

    # --- Single-file upload limits --------------------------------------------
    max_file_size_mb: int = 50
    max_upload_mb: int = 5
    max_demo_pdf_pages: int = 30

    # --- Cost / abuse guardrails (kept; also the Bedrock cost ceiling in Part 2)
    api_key: Optional[str] = None        # owner key -> bypasses all limits
    free_queries_per_day: int = 50       # questions per visitor (per IP) per day
    free_uploads_per_day: int = 10       # uploads per visitor (per IP) per day
    global_daily_cap: int = 1000         # hard ceiling across all visitors per day

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
