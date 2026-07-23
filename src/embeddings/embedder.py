from pathlib import Path

import duckdb

from .api import get_embedding

DB_PATH = (

    Path(__file__).resolve().parent.parent

    / "data"

    / "llm_ethics_data.duckdb"

)

def run_embeddings(response_table: str, embedding_table: str):

    con = duckdb.connect(str(DB_PATH))

    try:
        responses = con.execute(
            f"""
            SELECT r.response_id, r.response_text
            FROM {response_table} AS r
            LEFT JOIN {embedding_table} AS e
                ON r.response_id = e.response_id
            WHERE e.response_id IS NULL
            ORDER BY r.response_id
            """
        ).fetchall()

        total = len(responses)

        if total == 0:
            print(f"No new responses to embed for {response_table}.")
            return

        for index, (response_id, response_text) in enumerate(responses, start=1):
            embedding = get_embedding(response_text)

            con.execute(
                f"""
                INSERT INTO {embedding_table} (
                    response_id,
                    embedding
                )
                VALUES (?, ?)
                """,
                [
                    response_id,
                    embedding,
                ],
            )
        
            print(f"[{index}/{total}] Embedded response {response_id}")

    finally:
        con.close()

def main():
    run_embeddings("consistency_responses", "consistency_embeddings")
    run_embeddings("integrity_responses", "integrity_embeddings")

if __name__ == "__main__":
    main()