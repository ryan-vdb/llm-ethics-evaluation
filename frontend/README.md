# Results frontend

This dependency-free dashboard turns the canonical consistency and integrity
analysis snapshots into an interactive, presentation-ready site. Charts are
native SVG and every displayed number is generated from the committed JSON
results under `src/analysis`.

## Run locally

From this directory:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). No package installation,
API key, database write, or external network connection is required.

## Commands

```bash
npm run data      # refresh public/data/dashboard.json from both analyses
npm test          # validate the frontend data contract
npm run build     # create a deployable static site in dist/
npm run preview   # serve dist/ at :3000
npm run check     # sync data, test, and build
```

The generated `public/data/dashboard.json` is intentionally committed so the
site remains viewable as a static artifact. Run `npm run data` after rerunning
either Python analysis. `dist/` is a disposable build directory and is ignored
by Git.

## Structure

```text
frontend/
├── index.html
├── package.json
├── server.mjs
├── public/data/dashboard.json
├── scripts/
│   ├── build.mjs
│   └── sync-data.mjs
├── src/
│   ├── app.js
│   ├── charts.js
│   └── styles.css
└── tests/data-contract.test.mjs
```

The frontend deliberately preserves the analyses' claim boundaries. The
consistency result is framed as strong fixed-panel geometric evidence; the
integrity result is framed as modest, heterogeneous semantic responsiveness,
not verified conclusion reversal.
