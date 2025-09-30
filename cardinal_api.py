import os
import json
import pickle
import uuid
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from sentence_transformers import SentenceTransformer
import faiss

# Configuration for Cardinal Knowledge Base
KNOWLEDGE_BASE_DIR = "data"
NODES_FILE = f"{KNOWLEDGE_BASE_DIR}/nodes.bin"
EDGES_FILE = f"{KNOWLEDGE_BASE_DIR}/edges.bin"
NODE_INDEX_FILE = f"{KNOWLEDGE_BASE_DIR}/node_index.pkl" # For mapping IDs to file offsets
EDGE_INDEX_FILE = f"{KNOWLEDGE_BASE_DIR}/edge_index.pkl" # For mapping IDs to file offsets

VECTOR_INDEX_FILE = f"{KNOWLEDGE_BASE_DIR}/cardinal_vector_index.faiss" # Or .ann for Annoy
MAX_SIZE_GB = 4
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384 # Dimension for all-MiniLM-L6-v2 (for MiniLM-L6-v2)

@dataclass
class Node:
    id: str
    type: str
    content: str
    embedding: np.ndarray = field(repr=False) # repr=False to avoid printing large array
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For forgetting mechanism
    usage_count: int = 0
    timestamp_last_accessed: float = time.time()
    timestamp_created: float = time.time()
    connectivity: int = 0 # Number of edges connected to this node

@dataclass
class Edge:
    id: str
    source_node_id: str
    target_node_id: str
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CardinalKnowledgeBase:
    def __init__(self):
        os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embedding_dim = EMBEDDING_DIM

        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.faiss_index: Optional[faiss.IndexFlatL2] = None

        # Mappings for Faiss internal IDs to Node IDs and vice-versa
        self._faiss_id_to_node_id: Dict[int, str] = {}
        self._node_id_to_faiss_id: Dict[str, int] = {}

        self._load_state()

    def _load_state(self):
        # Load nodes and edges
        if os.path.exists(NODES_FILE):
            with open(NODES_FILE, 'rb') as f:
                self.nodes = pickle.load(f)

        if os.path.exists(EDGES_FILE):
            with open(EDGES_FILE, 'rb') as f:
                self.edges = pickle.load(f)
        
        # Rebuild Faiss index from loaded nodes to ensure consistency
        print("Rebuilding Faiss index from loaded nodes...")
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self._faiss_id_to_node_id = {}
        self._node_id_to_faiss_id = {}

        if self.nodes:
            all_embeddings = []
            for i, node_id in enumerate(self.nodes.keys()):
                node = self.nodes[node_id]
                all_embeddings.append(node.embedding)
                self._faiss_id_to_node_id[i] = node_id
                self._node_id_to_faiss_id[node_id] = i
            self.faiss_index.add(np.array(all_embeddings))

        print(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges. Faiss index has {self.faiss_index.ntotal} items.")

    def _save_state(self):
        # Save nodes and edges
        with open(NODES_FILE, 'wb') as f:
            pickle.dump(self.nodes, f)

        with open(EDGES_FILE, 'wb') as f:
            pickle.dump(self.edges, f)

        # Faiss index is not directly saved with its internal ID mappings.
        # The _load_state method rebuilds it from the self.nodes dictionary,
        # ensuring consistency of internal Faiss IDs with our Node IDs.
        # So, we don't need to faiss.write_index here if _load_state rebuilds.
        # However, if we want to avoid rebuilding every time, we would save the index
        # and manage _faiss_id_to_node_id and _node_id_to_faiss_id separately.
        # For FlatL2 and simplicity, rebuilding is acceptable for now given the small KB size.
        # If we had a persistent Faiss index, we'd faiss.write_index(self.faiss_index, VECTOR_INDEX_FILE)
        # For now, we rely on the rebuild on load.
        print("Saved Cardinal KB nodes and edges. Faiss index will be rebuilt on next load.")


    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text, convert_to_tensor=False).astype(np.float32)

    def _get_current_size_bytes(self) -> int:
        total_size = 0
        for f_path in [NODES_FILE, EDGES_FILE, VECTOR_INDEX_FILE]:
            if os.path.exists(f_path):
                total_size += os.path.getsize(f_path)
        return total_size

    def _calculate_node_importance(self, node: Node) -> float:
        # Heuristic for node importance
        # Higher usage_count, higher connectivity -> more important
        # Older, less recently accessed -> less important
        time_decay_factor = 0.1 # Weight for time decay
        current_time = time.time()
        
        # Normalize time components (e.g., to days or weeks) to prevent them from dominating
        age_in_days = (current_time - node.timestamp_created) / (3600 * 24)
        recency_in_days = (current_time - node.timestamp_last_accessed) / (3600 * 24)

        # Simple weighted sum. Weights can be tuned.
        # Usage and connectivity are positive, age and recency are negative influences.
        score = (node.usage_count * 1.0) + \
                (node.connectivity * 0.5) - \
                (age_in_days * time_decay_factor * 0.1) - \
                (recency_in_days * time_decay_factor * 0.5)
        return score

    def _manage_size(self):
        current_size_bytes = self._get_current_size_bytes()
        max_size_bytes = MAX_SIZE_GB * (1024**3)
        pruning_threshold_bytes = max_size_bytes * 0.9 # Start pruning if 90% full
        target_size_bytes = max_size_bytes * 0.8 # Prune until 80% full

        if current_size_bytes < pruning_threshold_bytes:
            # print(f"Current size {current_size_bytes / (1024**3):.2f} GB is below pruning threshold.")
            return

        print(f"Cardinal KB size exceeding threshold. Current: {current_size_bytes / (1024**3):.2f} GB, Max: {MAX_SIZE_GB:.2f} GB. Initiating forgetting mechanism...")

        # Calculate importance for all nodes
        node_scores = {
            node_id: self._calculate_node_importance(node)
            for node_id, node in self.nodes.items()
        }

        # Sort nodes by importance (ascending for deletion)
        sorted_nodes_to_prune = sorted(node_scores.items(), key=lambda item: item[1])

        nodes_deleted_count = 0
        for node_id, score in sorted_nodes_to_prune:
            if current_size_bytes <= target_size_bytes:
                print(f"Target size {target_size_bytes / (1024**3):.2f} GB reached. Stopped pruning.")
                break
            
            # Delete the node (which also handles edges and Faiss conceptually)
            if self.delete_knowledge(node_id): # delete_knowledge also calls _save_state implicitly
                nodes_deleted_count += 1
                # Recalculate size after deletion. This can be slow if done in loop.
                # A more optimized approach would estimate size reduction or delete in batches.
                current_size_bytes = self._get_current_size_bytes()
                print(f"  Deleted node {node_id} (score: {score:.2f}). New size: {current_size_bytes / (1024**3):.2f} GB")
            else:
                print(f"  Failed to delete node {node_id}.")
        
        if nodes_deleted_count > 0:
            print(f"Forgetting mechanism completed. Deleted {nodes_deleted_count} nodes. Final size: {current_size_bytes / (1024**3):.2f} GB.")
        else:
            print("Forgetting mechanism ran but no nodes were deleted (perhaps no nodes or already below threshold).")

        self._save_state() # Ensure final state is saved after all deletions

    def add_knowledge(self, content: str, node_type: str = "concept", metadata: Optional[Dict[str, Any]] = None) -> Node:
        node_id = str(uuid.uuid4())
        embedding = self._get_embedding(content)
        
        if metadata is None:
            metadata = {}

        new_node = Node(
            id=node_id,
            type=node_type,
            content=content,
            embedding=embedding,
            metadata=metadata
        )

        self.nodes[node_id] = new_node

        # Add embedding to Faiss index
        faiss_internal_id = self.faiss_index.ntotal # Get the next available internal ID
        self.faiss_index.add(np.array([embedding]))
        # Update mapping from Faiss internal ID to Node ID
        self._faiss_id_to_node_id[faiss_internal_id] = node_id
        self._node_id_to_faiss_id[node_id] = faiss_internal_id

        self._save_state()
        self._manage_size() # Check size after adding new knowledge
        return new_node

    def update_knowledge(self, node_id: str, new_content: Optional[str] = None, new_metadata: Optional[Dict[str, Any]] = None) -> Optional[Node]:
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]

        if new_content is not None and new_content != node.content:
            node.content = new_content
            node.embedding = self._get_embedding(new_content) # Re-embed if content changes
            
            # For FlatL2, updating an embedding effectively means replacing it.
            # Since _load_state rebuilds the index, we just need to ensure the node's embedding
            # in self.nodes is correct, and the next load will pick up the change.
            # For a more dynamic system without frequent reloads, faiss.IndexIDMap is better.
            print(f"Warning: Faiss index update for node {node_id} is a logical update. Physical deletion/re-addition not implemented for FlatL2 during runtime.")
            # The current approach relies on _load_state to rebuild the index correctly.
            # No need to modify faiss_index directly here as it will be rebuilt on next load.
            
        if new_metadata is not None:
            node.metadata.update(new_metadata)
        
        node.timestamp_last_accessed = time.time()
        self._save_state()
        return node

    def add_relationship(self, source_id: str, target_id: str, rel_type: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Edge]:
        if source_id not in self.nodes or target_id not in self.nodes:
            print(f"Error: Source node {source_id} or target node {target_id} not found.")
            return None
        
        edge_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}

        new_edge = Edge(
            id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            type=rel_type,
            metadata=metadata
        )

        self.edges[edge_id] = new_edge
        self.nodes[source_id].connectivity += 1
        self.nodes[target_id].connectivity += 1
        self._save_state()
        return new_edge

    def merge_knowledge(self, node_ids: List[str], new_content: str, new_type: str = "merged_concept", new_metadata: Optional[Dict[str, Any]] = None) -> Optional[Node]:
        if not node_ids or any(nid not in self.nodes for nid in node_ids):
            print("Error: One or more node IDs not found for merging.")
            return None
        
        # Create a new node from the merged content
        merged_node = self.add_knowledge(new_content, new_type, new_metadata)
        if merged_node is None:
            return None # Should not happen if add_knowledge works
        
        # Re-point all edges from old nodes to the new merged node
        nodes_to_delete_ids = set(node_ids)
        edges_to_update = []
        edges_to_delete = []

        for edge_id, edge in list(self.edges.items()): # Iterate over a copy because we're modifying
            if edge.source_node_id in nodes_to_delete_ids:
                if edge.target_node_id in nodes_to_delete_ids: # Edge between two nodes being merged
                    edges_to_delete.append(edge_id)
                else:
                    edges_to_update.append((edge_id, 'source'))
            elif edge.target_node_id in nodes_to_delete_ids:
                edges_to_update.append((edge_id, 'target'))
        
        for edge_id, role in edges_to_update:
            if role == 'source':
                self.edges[edge_id].source_node_id = merged_node.id
            else:
                self.edges[edge_id].target_node_id = merged_node.id
            self.nodes[merged_node.id].connectivity += 1 # New connection

        for edge_id in edges_to_delete:
            del self.edges[edge_id]

        # Delete the old nodes and their associated embeddings from Faiss (conceptually)
        for old_node_id in nodes_to_delete_ids:
            # Decrement connectivity for any nodes they were connected to
            for edge in self.edges.values():
                if edge.source_node_id == old_node_id:
                    if edge.target_node_id in self.nodes: self.nodes[edge.target_node_id].connectivity -= 1
                elif edge.target_node_id == old_node_id:
                    if edge.source_node_id in self.nodes: self.nodes[edge.source_node_id].connectivity -= 1

            # Remove node from our store. Faiss index will be rebuilt on load anyway.
            del self.nodes[old_node_id]
            
            # Remove from internal ID mappings. The Faiss index itself will be rebuilt on load.
            faiss_internal_id = self._node_id_to_faiss_id.pop(old_node_id, None)
            if faiss_internal_id is not None:
                del self._faiss_id_to_node_id[faiss_internal_id]

        self._save_state()
        print(f"Merged nodes {node_ids} into new node {merged_node.id}")
        return merged_node

    def consult(self, query: str, top_k: int = 5, association_depth: int = 1) -> List[Dict[str, Any]]:
        query_embedding = self._get_embedding(query)

        # Perform semantic search
        D, I = self.faiss_index.search(np.array([query_embedding]), top_k)
        
        results = []
        seen_nodes = set()

        # Get initial semantic matches
        for distance, faiss_id in zip(D[0], I[0]):
            if faiss_id == -1: # No result
                continue
            node_id = self._faiss_id_to_node_id.get(faiss_id)
            if node_id and node_id not in seen_nodes:
                node = self.nodes.get(node_id)
                if node:
                    node.usage_count += 1
                    node.timestamp_last_accessed = time.time()
                    # Convert L2 distance to a similarity score (0 to 1)
                    # Max L2 distance for normalized embeddings is 2.0 (when embeddings are opposite)
                    # So, 1 - (distance_L2^2 / 4) or 1 - (distance_L2 / 2) can be used.
                    # Using 1 - (distance / 2) assuming embeddings are normalized to unit sphere
                    similarity = 1 - (distance / 2) if distance >= 0 else 0
                    results.append({"node": node, "similarity": similarity, "source": "semantic_search"})
                    seen_nodes.add(node_id)

        # Traverse graph for associated knowledge
        associated_nodes = set()
        current_level_nodes = {r['node'].id for r in results}

        for _ in range(association_depth):
            next_level_nodes = set()
            for node_id in current_level_nodes:
                for edge in self.edges.values():
                    connected_node_id = None
                    if edge.source_node_id == node_id and edge.target_node_id not in seen_nodes and edge.target_node_id not in associated_nodes:
                        connected_node_id = edge.target_node_id
                    elif edge.target_node_id == node_id and edge.source_node_id not in seen_nodes and edge.source_node_id not in associated_nodes:
                        connected_node_id = edge.source_node_id
                    
                    if connected_node_id and connected_node_id in self.nodes:
                        connected_node = self.nodes[connected_node_id]
                        connected_node.usage_count += 1
                        connected_node.timestamp_last_accessed = time.time()
                        associated_nodes.add(connected_node_id)
                        next_level_nodes.add(connected_node_id)
                        results.append({"node": connected_node, "similarity": 0.0, "source": "association"}) # Associated nodes have 0 semantic similarity here
            current_level_nodes = next_level_nodes
            if not current_level_nodes: # No more associations to find
                break
        
        # Sort results (semantic matches first, then associated, potentially by connectivity/importance)
        # Note: Semantic matches will have similarity > 0, associated will have similarity 0 and association=True
        # We want to prioritize actual semantic matches first, then associated ones based on node importance.
        results.sort(key=lambda x: (x.get('similarity', 0), self._calculate_node_importance(x['node'])), reverse=True)
        
        self._save_state() # Save usage counts
        return results[:top_k] # Return only top_k after potential associations

    def get_knowledge_by_id(self, node_id: str) -> Optional[Node]:
        node = self.nodes.get(node_id)
        if node:
            node.usage_count += 1
            node.timestamp_last_accessed = time.time()
            self._save_state()
        return node

    def delete_knowledge(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        
        # Remove node from our store
        node_to_delete = self.nodes.pop(node_id)

        # Remove its embedding from Faiss (conceptually)
        # For FlatL2, actual deletion is complex without IndexIDMap.
        # We rely on _load_state to rebuild the index without this node.
        faiss_internal_id = self._node_id_to_faiss_id.pop(node_id, None)
        if faiss_internal_id is not None:
            del self._faiss_id_to_node_id[faiss_internal_id]
            # print(f"  Node {node_id} (Faiss ID {faiss_internal_id}) conceptually removed from Faiss mappings.")

        # Remove associated edges and update connectivity of connected nodes
        edges_to_remove = [eid for eid, edge in self.edges.items() if edge.source_node_id == node_id or edge.target_node_id == node_id]
        for eid in edges_to_remove:
            edge = self.edges.pop(eid)
            if edge.source_node_id != node_id and edge.source_node_id in self.nodes:
                self.nodes[edge.source_node_id].connectivity -= 1
            if edge.target_node_id != node_id and edge.target_node_id in self.nodes:
                self.nodes[edge.target_node_id].connectivity -= 1

        self._save_state()
        print(f"Deleted node {node_id} and {len(edges_to_remove)} associated edges.")
        return True

    def get_status(self) -> Dict[str, Any]:
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)
        faiss_count = self.faiss_index.ntotal if self.faiss_index else 0
        
        estimated_size_bytes = self._get_current_size_bytes()

        return {
            "nodes_count": total_nodes,
            "edges_count": total_edges,
            "faiss_index_count": faiss_count,
            "estimated_disk_usage_bytes": estimated_size_bytes,
            "max_size_gb": MAX_SIZE_GB,
            "embedding_model": EMBEDDING_MODEL_NAME
        }
