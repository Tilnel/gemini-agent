# cardinal.py
import json
import os
import time
import uuid
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import psycopg2
import psycopg2.pool
from bs4 import BeautifulSoup
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer


# --- Data Classes (remain the same) ---
@dataclass
class Node:
    id: str
    type: str
    content: str
    embedding: np.ndarray = field(repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    timestamp_last_accessed: float = time.time()
    timestamp_created: float = time.time()
    connectivity: int = 0

@dataclass
class Edge:
    id: str
    source_node_id: str
    target_node_id: str
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CardinalKnowledgeBase:
    def __init__(self, db_config: Dict[str, Any], agent_config: Dict[str, Any]):
        self.db_config = db_config
        self.agent_config = agent_config
        self.db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, **db_config)

        self.embedding_model = SentenceTransformer(agent_config['embedding_model'])
        self.embedding_dim = agent_config['embedding_dim']

        self.faiss_index: Optional[faiss.Index] = None
        self._faiss_id_to_node_id: Dict[int, str] = {}
        self._node_id_to_faiss_id: Dict[str, int] = {}

        self.KNOWLEDGE_BASE_DIR = "data"
        self.VECTOR_INDEX_FILE = f"{self.KNOWLEDGE_BASE_DIR}/cardinal_vector_index.faiss"
        os.makedirs(self.KNOWLEDGE_BASE_DIR, exist_ok=True)

        self._load_state()

    def _get_conn(self):
        return self.db_pool.getconn()

    def _put_conn(self, conn):
        self.db_pool.putconn(conn)

    def _load_state(self):
        """Load Faiss index from file or rebuild it from the database."""
        if os.path.exists(self.VECTOR_INDEX_FILE):
            print("Loading Faiss index from file...")
            self.faiss_index = faiss.read_index(self.VECTOR_INDEX_FILE)
            self._rebuild_faiss_mappings_from_db()
        else:
            print("No Faiss index file found. Rebuilding from database...")
            self._rebuild_faiss_index_from_db()

        if self.faiss_index:
            print(f"Cardinal KB (DB backend) is ready. Faiss index has {self.faiss_index.ntotal} items.")
        else:
            print("Cardinal KB is ready, but Faiss index is empty.")

    def _rebuild_faiss_mappings_from_db(self):
        """Re-populates the ID mappings from the DB, assuming the Faiss index is loaded."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM nodes ORDER BY timestamp_created;") # Ensure consistent order
                all_node_ids = [str(row[0]) for row in cur.fetchall()]
                # This simple mapping assumes the order in the DB matches the Faiss index.
                # A more robust solution might store faiss_id in the DB.
                self._faiss_id_to_node_id = {i: node_id for i, node_id in enumerate(all_node_ids)}
                self._node_id_to_faiss_id = {node_id: i for i, node_id in enumerate(all_node_ids)}
        finally:
            self._put_conn(conn)

    def _rebuild_faiss_index_from_db(self):
        """Fetches all embeddings from the database and builds a new Faiss index."""
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self._faiss_id_to_node_id.clear()
        self._node_id_to_faiss_id.clear()

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, embedding FROM nodes ORDER BY timestamp_created;")
                rows = cur.fetchall()
                if not rows:
                    print("No nodes in DB to build index from.")
                    return

                embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
                node_ids = [str(row[0]) for row in rows]

                self.faiss_index.add(embeddings)
                self._faiss_id_to_node_id = {i: node_id for i, node_id in enumerate(node_ids)}
                self._node_id_to_faiss_id = {node_id: i for i, node_id in enumerate(node_ids)}

                faiss.write_index(self.faiss_index, self.VECTOR_INDEX_FILE)
                print(f"Rebuilt and saved Faiss index with {len(rows)} vectors.")
        finally:
            self._put_conn(conn)

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode([text], convert_to_numpy=True)[0].astype(np.float32)

    def _clean_html_and_extra_whitespace(self, text: str) -> str:
        """Strips HTML tags and reduces multiple whitespace characters to a single space."""
        # Check if the content likely contains HTML
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, "lxml")
            text = soup.get_text(separator=' ', strip=True)
        # Replace multiple whitespace characters (including newlines, tabs) with a single space
        return ' '.join(text.split())

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Splits text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        return chunks

    def add_knowledge(self, content: str, node_type: str = "chunk", metadata: Optional[Dict[str, Any]] = None,
                      chunk_size: int = 1000, chunk_overlap: int = 150, auto_link_chunks: bool = True) -> List[Node]:
        """
        Cleans, chunks, and adds a long piece of content to the knowledge base.

        Args:
            content: The raw text or HTML content.
            node_type: The type of node for each chunk (default: "chunk").
            metadata: Optional metadata to associate with the parent document.
            chunk_size: The target size of each text chunk in characters.
            chunk_overlap: The number of characters to overlap between chunks.
            auto_link_chunks: If True, creates sequential relationships between chunks.

        Returns:
            A list of Node objects that were created and added.
        """
        # 1. Clean the incoming content
        cleaned_content = self._clean_html_and_extra_whitespace(content)
        if not cleaned_content:
            print("Warning: Content was empty after cleaning. No nodes were added.")
            return []

        # 2. Chunk the cleaned content
        text_chunks = self._chunk_text(cleaned_content, chunk_size, chunk_overlap)

        parent_doc_id = str(uuid.uuid4())
        created_nodes: List[Node] = []

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                for i, chunk in enumerate(text_chunks):
                    # 3. For each chunk, create a node
                    node_id = str(uuid.uuid4())
                    embedding = self._get_embedding(chunk)

                    chunk_metadata = (metadata or {}).copy()
                    chunk_metadata.update({
                        "parent_document_id": parent_doc_id,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    })

                    new_node = Node(
                        id=node_id, type=node_type, content=chunk,
                        embedding=embedding, metadata=chunk_metadata
                    )

                    # 4. Insert node into the database
                    cur.execute(
                        """
                        INSERT INTO nodes (id, type, content, embedding, metadata, timestamp_created)
                        VALUES (%s, %s, %s, %s, %s, %s);
                        """,
                        (node_id, node_type, chunk, embedding.tobytes(), Json(chunk_metadata), datetime.datetime.fromtimestamp(new_node.timestamp_created, tz=datetime.timezone.utc))
                    )
                    created_nodes.append(new_node)

            conn.commit()
            print(f"Added {len(created_nodes)} chunks to DB for document {parent_doc_id}.")

            # 5. Optionally, link the chunks sequentially
            if auto_link_chunks and len(created_nodes) > 1:
                for i in range(len(created_nodes) - 1):
                    source_node = created_nodes[i]
                    target_node = created_nodes[i+1]
                    self.add_relationship(source_node.id, target_node.id, "is_followed_by")
                print(f"Created {len(created_nodes) - 1} sequential relationships.")

        except psycopg2.Error as e:
            print(f"Database error during knowledge addition: {e}")
            conn.rollback()
            return [] # Return empty list on failure
        finally:
            self._put_conn(conn)

        # For simplicity, we still rely on a periodic or startup rebuild of Faiss.
        # A more advanced implementation would add this to the index in real-time.
        print("Faiss index will update on next restart. Rebuilding now for immediate use...")
        self._rebuild_faiss_index_from_db() # Added for immediate usability

        return created_nodes

    def add_relationship(self, source_id: str, target_id: str, rel_type: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Edge]:
        edge_id = uuid.uuid4()
        new_edge = Edge(str(edge_id), source_id, target_id, rel_type, metadata or {})

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO edges (id, source_node_id, target_node_id, type, metadata)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (str(edge_id), source_id, target_id, rel_type, Json(metadata or {}))
                )
                cur.execute("UPDATE nodes SET connectivity = connectivity + 1 WHERE id IN (%s, %s);", (source_id, target_id))
                conn.commit()
        except psycopg2.Error as e:
            print(f"Database error adding relationship: {e}")
            conn.rollback()
            return None
        finally:
            self._put_conn(conn)

        return new_edge

    def consult(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            print("Cannot consult: Faiss index is empty.")
            return []

        query_embedding = self._get_embedding(query)
        D, I = self.faiss_index.search(np.array([query_embedding]), top_k)

        faiss_ids = I[0]
        distances = D[0]

        # Filter out invalid Faiss IDs (-1)
        valid_indices = [i for i, fid in enumerate(faiss_ids) if fid != -1]
        if not valid_indices:
            return []

        initial_node_ids = [self._faiss_id_to_node_id[faiss_ids[i]] for i in valid_indices]

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Using tuple for IN clause for performance
                cur.execute("SELECT id, type, content, metadata FROM nodes WHERE id IN %s;", (tuple(initial_node_ids),))
                rows = cur.fetchall()

                # Map results for easy lookup and to preserve Faiss order
                node_data_map = {str(row[0]): {'type': row[1], 'content': row[2], 'metadata': row[3]} for row in rows}

                results = []
                for i in valid_indices:
                    faiss_id = faiss_ids[i]
                    distance = distances[i]
                    node_id = self._faiss_id_to_node_id.get(faiss_id)

                    if node_id and node_id in node_data_map:
                        data = node_data_map[node_id]
                        # L2 distance is squared, similarity can be represented in various ways.
                        # 1 / (1 + L2_distance) is a common one.
                        similarity = 1 / (1 + distance)

                        results.append({
                            "id": node_id,
                            "content": data['content'],
                            "type": data['type'],
                            "metadata": data['metadata'],
                            "similarity": similarity,
                            "source": "semantic_search"
                        })
                return results
        finally:
            self._put_conn(conn)

    def close(self):
        """Closes the database connection pool."""
        if self.db_pool:
            self.db_pool.closeall()
            print("Database connection pool closed.")
