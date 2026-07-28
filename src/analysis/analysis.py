from itertools import combinations
from pathlib import Path

import duckdb

from .api import cosine_similarity

DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "llm_ethics_data.duckdb"
)


def analyse_embeddings(
    response_table: str,
    embedding_table: str,
):

    con = duckdb.connect(str(DB_PATH))

    try:

        rows = con.execute(
            f"""
            SELECT
                r.response_id,
                r.model,
                r.question_id,
                e.embedding
            FROM {response_table} r
            JOIN {embedding_table} e
                ON r.response_id = e.response_id
            ORDER BY r.question_id, r.model
            """
        ).fetchall()

        print(f"\nLoaded {len(rows)} embeddings.")

        ####################################################
        # ACROSS MODELS
        ####################################################

        print("\n========== ACROSS MODELS ==========\n")

        pair_scores = {}

        for i in range(len(rows)):

            _, model1, question1, emb1 = rows[i]

            for j in range(i + 1, len(rows)):

                _, model2, question2, emb2 = rows[j]

                if question1 != question2:
                    continue

                score = cosine_similarity(emb1, emb2)

                pair = tuple(sorted((model1, model2)))

                pair_scores.setdefault(pair, []).append(score)

        for pair in sorted(pair_scores):

            scores = pair_scores[pair]

            print(
                f"{pair[0]} vs {pair[1]}:"
                f" Average = {sum(scores)/len(scores):.4f}"
                f"  Min = {min(scores):.4f}"
                f"  Max = {max(scores):.4f}"
            )

        ####################################################
        # WITHIN MODELS
        ####################################################

        print("\n========== WITHIN MODEL ==========\n")

        model_embeddings = {}

        for _, model, question, embedding in rows:
            model_embeddings.setdefault(model, []).append(
                (question, embedding)
            )

        for model in sorted(model_embeddings):

            embeddings = model_embeddings[model]

            comparisons = []

            for (q1, emb1), (q2, emb2) in combinations(embeddings, 2):

                score = cosine_similarity(emb1, emb2)

                comparisons.append((score, q1, q2))

            avg = sum(score for score, _, _ in comparisons) / len(comparisons)

            min_score, min_q1, min_q2 = min(
                comparisons,
                key=lambda x: x[0],
            )

            max_score, max_q1, max_q2 = max(
                comparisons,
                key=lambda x: x[0],
            )

            print(f"\n{model}")
            print(f"Average = {avg:.4f}")
            print(
                f"Minimum = {min_score:.4f} "
                f"(Questions {min_q1} & {min_q2})"
            )
            print(
                f"Maximum = {max_score:.4f} "
                f"(Questions {max_q1} & {max_q2})"
            )

    finally:
        con.close()


def main():

    print("\n==============================")
    print("CONSISTENCY")
    print("==============================")

    analyse_embeddings(
        "consistency_responses",
        "consistency_embeddings",
    )

    print("\n==============================")
    print("INTEGRITY")
    print("==============================")

    analyse_embeddings(
        "integrity_responses",
        "integrity_embeddings",
    )


if __name__ == "__main__":
    main()