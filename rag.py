import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'

def chunk_dataframe(df: pd.DataFrame):
    chunks = []
    for idx, row in df.iterrows():
        chunk = f"Row {idx}: " + ", ".join([f"{col}={row[col]}" for col in df.columns])
        chunks.append(chunk)
    return chunks

def build_faiss_index(df: pd.DataFrame):
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    chunks = chunk_dataframe(df)
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks, embedder

def retrieve_relevant_chunks(query, index, chunks, embedder, top_k=3):
    query_emb = embedder.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), top_k)
    return [chunks[i] for i in I[0]]