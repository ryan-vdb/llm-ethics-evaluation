import duckdb

con = duckdb.connect("llm_ethics_data.duckdb")

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
            
""")