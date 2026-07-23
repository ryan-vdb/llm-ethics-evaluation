from pathlib import Path

import duckdb

from .api import get_embedding

DB_PATH = (

    Path(__file__).resolve().parent.parent

    / "data"

    / "llm_ethics_data.duckdb"

)


def get_next_response_id(connection: duckdb.DuckDBPyConnection) -> int:
    result = connection.execute(
        """
        SELECT COALESCE(MAX(response_id), 0) + 1
        FROM consistency_embeddings
        """
    ).fetchone()

    return result[0]


def run_embeddings(question_id: int):

    con = duckdb.connect(str(DB_PATH))

    try:
        question = con.execute(
            """
            SELECT question_text
            FROM consistency_questions
            WHERE question_id = ?
            """,
            [question_id],
        ).fetchone()

        if question is None:
            raise ValueError(
                f"Question {question_id} does not exist."
            )

        question_text = question[0]

        embedding = get_embedding(question_text)

        response_id = get_next_response_id(con)

        con.execute(
            """
            INSERT INTO consistency_embeddings (
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

        return embedding

    finally:
        con.close()


def main():

    con = duckdb.connect(str(DB_PATH))

    try:
        question_ids = [
            row[0]
            for row in con.execute(
                """
                SELECT question_id
                FROM consistency_questions
                ORDER BY question_id
                """
            ).fetchall()
        ]
    finally:
        con.close()

    total = len(question_ids)

    for index, question_id in enumerate(question_ids, start=1):

        run_embeddings(question_id)

        print(
            f"[{index}/{total}] Embedded question {question_id}"
        )


if __name__ == "__main__":
    main()