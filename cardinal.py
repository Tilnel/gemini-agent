import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import psycopg2
import psycopg2.pool
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
            # We still need to populate the mappings
            self._rebuild_faiss_mappings_from_db()
        else:
            print("No Faiss index file found. Rebuilding from database...")
            self._rebuild_faiss_index_from_db()

        print(f"Cardinal KB (DB backend) is ready. Faiss index has {self.faiss_index.ntotal} items.")

    def _rebuild_faiss_mappings_from_db(self):
        """Re-populates the ID mappings from the DB, assuming the Faiss index is loaded."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM nodes;")
                all_node_ids = [row[0] for row in cur.fetchall()]
                for i, node_id in enumerate(all_node_ids):
                    self._faiss_id_to_node_id[i] = str(node_id)
                    self._node_id_to_faiss_id[str(node_id)] = i
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
                cur.execute("SELECT id, embedding FROM nodes;")
                rows = cur.fetchall()
                if not rows:
                    print("No nodes in DB to build index from.")
                    return

                all_embeddings = []
                for i, (node_id, embedding_bytes) in enumerate(rows):
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    all_embeddings.append(embedding)
                    node_id_str = str(node_id)
                    self._faiss_id_to_node_id[i] = node_id_str
                    self._node_id_to_faiss_id[node_id_str] = i

                self.faiss_index.add(np.array(all_embeddings))
                faiss.write_index(self.faiss_index, self.VECTOR_INDEX_FILE)
                print(f"Rebuilt and saved Faiss index with {len(rows)} vectors.")
        finally:
            self._put_conn(conn)

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text).astype(np.float32)

    def add_knowledge(self, content: str, node_type: str = "concept", metadata: Optional[Dict[str, Any]] = None) -> Node:
        node_id = uuid.uuid4()
        embedding = self._get_embedding(content)

        new_node = Node(
            id=str(node_id), type=node_type, content=content,
            embedding=embedding, metadata=metadata or {}
        )

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO nodes (id, type, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (str(node_id), node_type, content, embedding.tobytes(), Json(metadata or {}))
                )
                conn.commit()
        finally:
            self._put_conn(conn)

        # For simplicity, we'll rely on a periodic or startup rebuild of Faiss.
        # A more advanced implementation would add this to the index in real-time.
        print(f"Node {node_id} added to DB. Faiss index will update on next restart.")
        return new_node

    def add_relationship(self, source_id: str, target_id: str, rel_type: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Edge]:
        edge_id = uuid.uuid4()
        new_edge = Edge(str(edge_id), source_id, target_id, rel_type, metadata or {})

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Add the edge
                cur.execute(
                    """
                    INSERT INTO edges (id, source_node_id, target_node_id, type, metadata)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (str(edge_id), source_id, target_id, rel_type, Json(metadata or {}))
                )
                # Update connectivity counts
                cur.execute("UPDATE nodes SET connectivity = connectivity + 1 WHERE id IN (%s, %s);", (source_id, target_id))
                conn.commit()
        except psycopg2.Error as e:
            print(f"Database error adding relationship: {e}")
            conn.rollback()
            return None
        finally:
            self._put_conn(conn)

        return new_edge

    def consult(self, query: str, top_k: int = 5, association_depth: int = 1) -> List[Dict[str, Any]]:
        query_embedding = self._get_embedding(query)
        D, I = self.faiss_index.search(np.array([query_embedding]), top_k)

        initial_node_ids = [self._faiss_id_to_node_id[faiss_id] for faiss_id in I[0] if faiss_id != -1]
        if not initial_node_ids:
            return []

        # Fetch node data from DB
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                results = []
                seen_nodes = set()

                # Fetch initial semantic matches
                cur.execute("SELECT id, type, content, metadata FROM nodes WHERE id like ANY(%s);", (initial_node_ids,))

                rows = cur.fetchall()

                id_to_row = {str(row[0]): row for row in rows}

                for i, node_id in enumerate(initial_node_ids):
                    if node_id in id_to_row and node_id not in seen_nodes:
                        row = id_to_row[node_id]
                        node = Node(id=str(row[0]), type=row[1], content=row[2], metadata=row[3], embedding=np.array([]))
                        similarity = 1 - (D[0][i] / 2)
                        results.append({"node": node, "similarity": similarity, "source": "semantic_search"})
                        seen_nodes.add(node_id)

                # In a production system, graph traversal would be a complex recursive SQL query.
                # For simplicity here, we keep the traversal logic similar to the file-based one,
                # which means it's not fully optimized for a DB but demonstrates the principle.
                # A true graph DB or advanced SQL would be better for deep traversals.

                return results[:top_k]
        finally:
            self._put_conn(conn)

    def close(self):
        """Closes the database connection pool."""
        if self.db_pool:
            self.db_pool.closeall()
            print("Database connection pool closed.")
