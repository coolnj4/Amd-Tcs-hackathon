"""
Embeddings & ChromaDB — Manages embedding generation and vector store operations.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, EMBEDDING_DIM,
    CHROMA_DB_PATH, COMPLIANCE_COLLECTION_NAME, DOCUMENT_COLLECTION_PREFIX,
    RAG_TOP_K,
)


class EmbeddingManager:
    """Manages embedding model and ChromaDB collections."""

    def __init__(self, model_name=None, device=None, chroma_path=None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.device = device or EMBEDDING_DEVICE
        self.chroma_path = chroma_path or CHROMA_DB_PATH

        print(f"  Loading embedding model: {self.model_name} on {self.device}...")
        self.embed_model = SentenceTransformer(self.model_name, device=self.device)
        dim = EMBEDDING_DIM
        if hasattr(self.embed_model, "get_embedding_dimension"):
            dim = self.embed_model.get_embedding_dimension()
        elif hasattr(self.embed_model, "get_sentence_embedding_dimension"):
            dim = self.embed_model.get_sentence_embedding_dimension()
        print(f"  ✅ Embedding model loaded (dim={dim})")

        os.makedirs(self.chroma_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        print(f"  ✅ ChromaDB initialized at {self.chroma_path}")

    def embed(self, texts: list) -> list:
        """Generate embeddings for a list of texts."""
        # BGE models benefit from a prefix for retrieval
        prefixed = [f"Represent this document for retrieval: {t}" for t in texts]
        embeddings = self.embed_model.encode(prefixed, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list:
        """Generate embedding for a query."""
        prefixed = f"Represent this query for retrieval: {query}"
        embedding = self.embed_model.encode([prefixed], show_progress_bar=False)
        return embedding[0].tolist()

    # ─── Collection Management ────────────────────────────

    def get_or_create_compliance_collection(self):
        """Get or create the SEBI compliance collection."""
        return self.chroma_client.get_or_create_collection(
            name=COMPLIANCE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def get_or_create_document_collection(self, company_slug: str):
        """Get or create a document-specific collection."""
        name = f"{DOCUMENT_COLLECTION_PREFIX}{company_slug}"
        # ChromaDB collection names must be 3-63 chars, alphanumeric + _ -
        name = name[:63].replace(" ", "_").replace(".", "_")
        return self.chroma_client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_document_collection(self, company_slug: str):
        """Delete a document collection (cleanup)."""
        name = f"{DOCUMENT_COLLECTION_PREFIX}{company_slug}"
        name = name[:63].replace(" ", "_").replace(".", "_")
        try:
            self.chroma_client.delete_collection(name)
        except Exception:
            pass

    # ─── Indexing ─────────────────────────────────────────

    def index_chunks(self, collection, chunks: list, batch_size: int = 50):
        """
        Index a list of Chunk objects into a ChromaDB collection.

        Args:
            collection: ChromaDB collection
            chunks: List of Chunk objects
            batch_size: Number of chunks to process at once
        """
        total = len(chunks)
        indexed = 0

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.metadata for c in batch]

            # Generate embeddings
            embeddings = self.embed(texts)

            # Deduplicate IDs (ChromaDB requires unique IDs)
            seen_ids = set()
            unique_texts, unique_ids, unique_metas, unique_embeds = [], [], [], []
            for t, id_, m, e in zip(texts, ids, metadatas, embeddings):
                if id_ not in seen_ids:
                    seen_ids.add(id_)
                    unique_texts.append(t)
                    unique_ids.append(id_)
                    unique_metas.append(m)
                    unique_embeds.append(e)

            # Upsert into collection
            collection.upsert(
                documents=unique_texts,
                ids=unique_ids,
                metadatas=unique_metas,
                embeddings=unique_embeds,
            )
            indexed += len(unique_ids)

        print(f"  ✅ Indexed {indexed}/{total} chunks")

    # ─── Retrieval ────────────────────────────────────────

    def query(self, collection, query_text: str, top_k: int = None,
              where: dict = None) -> list:
        """
        Query a ChromaDB collection.

        Args:
            collection: ChromaDB collection
            query_text: Query string
            top_k: Number of results
            where: Optional metadata filter

        Returns:
            List of dicts with keys: text, metadata, distance
        """
        top_k = top_k or RAG_TOP_K
        query_embedding = self.embed_query(query_text)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count()) if collection.count() > 0 else 1,
        }
        if where:
            kwargs["where"] = where

        try:
            results = collection.query(**kwargs)
        except Exception as e:
            print(f"  [WARN] Query failed: {e}")
            return []

        # Parse results
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "id": results["ids"][0][i] if results["ids"] else "",
                })

        return output

    def list_collections(self) -> list:
        """List all ChromaDB collections."""
        return [c.name for c in self.chroma_client.list_collections()]
