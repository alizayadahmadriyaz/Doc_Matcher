import os
import shutil
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()


# --- Config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "website_data")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def download_faiss_dir(vector_store_path: str) -> str:
    """Download FAISS vector store from Supabase to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    bucket = supabase.storage.from_(SUPABASE_BUCKET)

    files = bucket.list(vector_store_path)
    if not files:
        raise FileNotFoundError(f"No files found in Supabase path: {vector_store_path}")

    for file in files:
        file_name = file["name"]
        file_path = os.path.join(temp_dir, file_name)
        res = bucket.download(f"{vector_store_path}/{file_name}")
        with open(file_path, "wb") as f:
            f.write(res)
    return temp_dir


def get_top_k_similar_contexts(vector_store_path: str, query: str, k: int = 5):
    """
    Retrieve top-k most similar contexts from FAISS vector store.
    Returns both distance (lower=better) and normalized similarity (higher=better).
    """
    local_dir = download_faiss_dir(vector_store_path)
    try:
        vectorstore = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search_with_score(query, k=k*2)  # get more, then sort manually

        # Convert distances to normalized similarities
        results_with_similarity = [
            (doc, float(score), round(1 / (1 + float(score)), 4))  # normalize
            for doc, score in results
        ]

        # Sort: lowest distance → highest similarity
        sorted_results = sorted(results_with_similarity, key=lambda x: x[1])[:k]

        output = [
            {
                "content": doc.page_content,
                "distance": distance,
                "similarity": similarity,
                "metadata": doc.metadata,
            }
            for doc, distance, similarity in sorted_results
        ]
        return output
    finally:
        shutil.rmtree(local_dir, ignore_errors=True)
