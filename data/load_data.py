from pathlib import Path

import duckdb


# Folder containing this script: project/data/
DATA_DIR = Path(__file__).resolve().parent

# Files inside data/
DB_PATH = DATA_DIR / "llm_ethics_data.duckdb"
CSV_DIR = DATA_DIR / "csvs"

CONSISTENCY_CSV = CSV_DIR / "consistency_questions.csv"
INTEGRITY_CSV = CSV_DIR / "integrity_questions.csv"
HELPERS_CSV = CSV_DIR / "helpers.csv"


# Check that the CSV files exist before changing the database
for csv_path in [CONSISTENCY_CSV, INTEGRITY_CSV, HELPERS_CSV]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")


con = duckdb.connect(str(DB_PATH))

try:
    # Clear and reload consistency questions
    con.execute("DELETE FROM consistency_questions;")

    con.execute(
        """
        INSERT INTO consistency_questions (
            question_id,
            domain,
            hidden_conflict,
            source,
            question_text
        )
        SELECT
            question_id,
            domain,
            hidden_conflict,
            source,
            question_text
        FROM read_csv_auto(?);
        """,
        [str(CONSISTENCY_CSV)],
    )

    # Clear and reload integrity questions
    con.execute("DELETE FROM integrity_questions;")

    con.execute(
        """
        INSERT INTO integrity_questions (
            question_id,
            domain,
            hidden_conflict,
            source,
            question_text
        )
        SELECT
            question_id,
            domain,
            hidden_conflict,
            source,
            question_text
        FROM read_csv_auto(?);
        """,
        [str(INTEGRITY_CSV)],
    )

    # Clear and reload helpers
    con.execute("DELETE FROM helpers;")

    con.execute(
        """
        INSERT INTO helpers (
            helper_type,
            helper_text
        )
        SELECT
            helper_type,
            helper_text
        FROM read_csv_auto(?);
        """,
        [str(HELPERS_CSV)],
    )

    print("Data loaded successfully.")

    # Show how many rows were loaded
    consistency_count = con.execute(
        "SELECT COUNT(*) FROM consistency_questions;"
    ).fetchone()[0]

    integrity_count = con.execute(
        "SELECT COUNT(*) FROM integrity_questions;"
    ).fetchone()[0]

    helper_count = con.execute(
        "SELECT COUNT(*) FROM helpers;"
    ).fetchone()[0]

    print(f"Consistency questions: {consistency_count}")
    print(f"Integrity questions: {integrity_count}")
    print(f"Helpers: {helper_count}")

finally:
    con.close()