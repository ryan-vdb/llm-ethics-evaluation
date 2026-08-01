# LLM Ethics Evaluation

The completed analyses and reproducible runners are documented here:

- [Cross-topic consistency analysis](src/analysis/consistency/README.md)
- [Integrity under social feedback](src/analysis/integrity/README.md)

## Local API keys

API keys are needed only for collecting new responses or embeddings. The
completed analyses read the existing DuckDB and do not require either key.

Install the root requirements in the environment used for data collection.
For a fresh checkout, copy the committed template and fill in the local file:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

```dotenv
OPENROUTER_GENERATION_API_KEY=your_generation_key
OPENROUTER_EMBEDDING_API_KEY=your_embedding_key
```

The clients load `.env` from the repository root. Existing shell environment
variables take precedence. The real `.env` and other `.env.*` variants are
ignored by Git; only `.env.example` is committed.
