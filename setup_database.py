import json
import psycopg2
from psycopg2 import sql

def create_tables():
    """
    Connects to the PostgreSQL database specified in config.json
    and creates the necessary 'nodes' and 'edges' tables for the knowledge base.
    """
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)['database']
    except FileNotFoundError:
        print("Error: config.json not found. Please create it with your database credentials.")
        return

    conn = None
    try:
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**config)
        cur = conn.cursor()

        # Drop existing tables if they exist to start fresh
        print("Dropping existing tables (if any)...")
        cur.execute("DROP TABLE IF EXISTS edges;")
        cur.execute("DROP TABLE IF EXISTS nodes;")

        # Create the 'nodes' table
        print("Creating 'nodes' table...")
        cur.execute("""
        CREATE TABLE nodes (
            id UUID PRIMARY KEY,
            type VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            embedding BYTEA NOT NULL,
            metadata JSONB,
            usage_count INTEGER DEFAULT 0,
            timestamp_created TIMESTAMPTZ DEFAULT NOW(),
            timestamp_last_accessed TIMESTAMPTZ DEFAULT NOW(),
            connectivity INTEGER DEFAULT 0
        );
        """)

        # Create the 'edges' table
        print("Creating 'edges' table...")
        cur.execute("""
        CREATE TABLE edges (
            id UUID PRIMARY KEY,
            source_node_id UUID NOT NULL,
            target_node_id UUID NOT NULL,
            type VARCHAR(255) NOT NULL,
            metadata JSONB,
            CONSTRAINT fk_source_node
                FOREIGN KEY(source_node_id) 
                REFERENCES nodes(id)
                ON DELETE CASCADE,
            CONSTRAINT fk_target_node
                FOREIGN KEY(target_node_id) 
                REFERENCES nodes(id)
                ON DELETE CASCADE
        );
        """)
        
        # Create indexes for faster lookups
        print("Creating indexes...")
        cur.execute("CREATE INDEX idx_edges_source ON edges (source_node_id);")
        cur.execute("CREATE INDEX idx_edges_target ON edges (target_node_id);")

        conn.commit()
        print("\nDatabase tables 'nodes' and 'edges' created successfully.")
        print("You can now run the main agent application.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while setting up PostgreSQL database: {error}")
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    create_tables()

