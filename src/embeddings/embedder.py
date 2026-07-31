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


def embed_questions() -> None:

    con = duckdb.connect(str(DB_PATH))

    questions = con.execute("""
        SELECT q.question_id, q.question_text
        FROM (
            SELECT question_id, question_text
            FROM consistency_questions

            UNION ALL

            SELECT question_id, question_text
            FROM integrity_questions
        ) AS q
        LEFT JOIN question_embeddings AS e
            ON q.question_id = e.question_id
        WHERE e.question_id IS NULL
        ORDER BY q.question_id
    """).fetchall()

    total = len(questions)

    if total == 0:
        print("All questions are already embedded.")
        return

    for index, (question_id, question_text) in enumerate(questions, start=1):
        embedding = get_embedding(question_text)

        con.execute(
            """
            INSERT INTO question_embeddings (
                question_id,
                embedding
            )
            VALUES (?, ?)
            """,
            [question_id, embedding],
        )

        print(f"Embedded question {index}/{total} (question_id={question_id})")

    print(f"Finished embedding {total} questions.")

def main():
    run_embeddings("consistency_responses", "consistency_embeddings")
    run_embeddings("integrity_responses", "integrity_embeddings")
    embed_questions()

if __name__ == "__main__":
    main()