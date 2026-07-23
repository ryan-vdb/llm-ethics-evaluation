from pathlib import Path
import duckdb

DATA_DIR = Path(__file__).resolve().parent
DB_PATH = DATA_DIR / "llm_ethics_data.duckdb"

con = duckdb.connect(str(DB_PATH))

con.execute("""

    CREATE TABLE IF NOT EXISTS consistency_questions (

        question_id INTEGER PRIMARY KEY,
            
        domain VARCHAR NOT NULL,
            
        hidden_conflict VARCHAR NOT NULL,
            
        source VARCHAR NOT NULL,

        question_text VARCHAR NOT NULL,
            
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
            
    CREATE TABLE IF NOT EXISTS integrity_questions (
            
        question_id INTEGER PRIMARY KEY,
            
        domain VARCHAR NOT NULL,
            
        hidden_conflict VARCHAR NOT NULL,
            
        source VARCHAR NOT NULL,

        question_text VARCHAR NOT NULL,
            
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS helpers (
            
        helper_type VARCHAR NOT NULL,

        helper_text VARCHAR NOT NULL
    );
            
    CREATE TABLE IF NOT EXISTS consistency_responses (

        response_id INTEGER PRIMARY KEY,
            
        model VARCHAR NOT NULL,
            
        question_id INTEGER NOT NULL,
            
        response_text VARCHAR NOT NULL,
            
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP        
    );
            
    CREATE TABLE IF NOT EXISTS integrity_responses (
            
        response_id INTEGER PRIMARY KEY,
        
        model VARCHAR NOT NULL,
        
        question_id INTEGER NOT NULL,
            
        helper_type VARCHAR,
            
        response_text VARCHAR NOT NULL,
            
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
            
    CREATE TABLE IF NOT EXISTS consistency_embeddings (
            
        response_id INTEGER PRIMARY KEY,

        embedding DOUBLE[] NOT NULL,
            
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
            
    CREATE TABLE IF NOT EXISTS integrity_embeddings (
            
        response_id INTEGER PRIMARY KEY,
            
        embedding DOUBLE[] NOT NULL,
            
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
            
""")

con.close()

print("Tables initialized successfully!")



