from itertools import combinations
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

from .linalg import cosine_similarity

DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "llm_ethics_data.duckdb"
)


def embeddings_table(
    response_table: str,
    embedding_table: str,
) -> pd.DataFrame:

    con = duckdb.connect(str(DB_PATH))

    try:

        df = con.execute(
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
            ).df()

        print(f"\nLoaded {len(df)} embeddings.")

        return df
    
    finally:
        con.close()

def similarity_matrices(embeddings_df: pd.DataFrame) -> dict[str, pd.DataFrame]:

    required_columns = {
        "response_id",
        "model",
        "question_id",
        "embedding",
    }

    missing_columns = required_columns - set(embeddings_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    
    matrices = {}

    sorted_df = embeddings_df.sort_values(
        by=["model", "question_id"]
    ).reset_index(drop=True)

    for model, model_df in sorted_df.groupby("model", sort=True):
        model_df = model_df.sort_values("question_id").reset_index(drop=True)
        
        question_ids = model_df["question_id"].tolist()

        embedding_matrix = np.vstack(
            model_df["embedding"].apply(
                lambda embedding: np.asarray(embedding, dtype=float)
            )
        )

        similarity_matrix = np.array([
            [
                cosine_similarity(embedding1, embedding2)
                for embedding2 in embedding_matrix
            ]
            for embedding1 in embedding_matrix
        ])

        matrices[model] = pd.DataFrame(
            similarity_matrix,
            index=question_ids,
            columns=question_ids,
        )
    
    return matrices

def aggregate_matrix(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:

    matrix_array = np.stack([
        matrix.to_numpy()
        for matrix in matrices.values()
    ])

    aggregate_values = matrix_array.mean(axis=0)

    first_matrix = next(iter(matrices.values()))

    return pd.DataFrame(
        aggregate_values,
        index=first_matrix.index,
        columns=first_matrix.columns
    )

def residual_matrices(matrices: dict[str, pd.DataFrame], aggregate: pd.DataFrame) -> dict[str, pd.DataFrame]:
    
    return{
        model : matrix - aggregate
        for model, matrix in matrices.items()
    }
   

        

        ####################################################
        # ACROSS MODELS
        ####################################################
    '''
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

        '''

def main():

    embeddings_df = embeddings_table(
        response_table="consistency_responses",
        embedding_table="consistency_embeddings",
    )

    matrices = similarity_matrices(embeddings_df)

    aggregate = aggregate_matrix(matrices)

    residuals = residual_matrices(
        matrices,
        aggregate,
    )

    print("\n" + "=" * 60)
    print("Similarity Analysis")
    print("=" * 60)

    print(f"\nModels: {len(matrices)}")
    print(f"Matrix shape: {aggregate.shape}")

    print("\nAggregate matrix — top-left 5×5:")
    print(aggregate.iloc[:5, :5].round(3))

    print("\n" + "=" * 60)
    print("Residual Matrices")
    print("=" * 60)

    for model, residual in residuals.items():

        print(f"\nModel: {model}")

        print(
            f"Residual range: "
            f"{residual.to_numpy().min():.3f} to "
            f"{residual.to_numpy().max():.3f}"
        )

        print("\nTop-left 5×5:")
        print(residual.iloc[:5, :5].round(3))


if __name__ == "__main__":
    main()