# LLM Ethics Evaluation

The completed analyses and reproducible runners are documented here:

- [Cross-topic consistency analysis](src/analysis/consistency/README.md)
- [Integrity under social feedback](src/analysis/integrity/README.md)

## Interactive results dashboard

The presentation-ready frontend combines both analyses into an interactive,
graphical explanation of the data, methods, evidence, and limitations.

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). It uses only committed
result snapshots, so viewing or rebuilding it does not require API keys.
See [frontend/README.md](frontend/README.md) for the data-refresh, test, and
static-build commands.

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
