import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = dirname(fileURLToPath(import.meta.url));
const argumentsList = process.argv.slice(2);

function argumentValue(name, fallback) {
  const index = argumentsList.indexOf(name);
  return index >= 0 && argumentsList[index + 1] ? argumentsList[index + 1] : fallback;
}

const port = Number(argumentValue("--port", process.env.PORT || "3000"));
const requestedRoot = argumentValue("--root", ".");
const serveRoot = resolve(frontendRoot, requestedRoot);
const development = serveRoot === frontendRoot;

if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error(`Invalid port: ${port}`);
}

const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".csv", "text/csv; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".md", "text/markdown; charset=utf-8"],
  [".svg", "image/svg+xml; charset=utf-8"],
  [".txt", "text/plain; charset=utf-8"],
  [".webmanifest", "application/manifest+json"],
]);

function safePath(root, pathname) {
  const decoded = decodeURIComponent(pathname).replaceAll("\\", "/");
  const candidate = resolve(root, `.${decoded}`);
  if (candidate !== root && !candidate.startsWith(`${root}${sep}`)) {
    return null;
  }
  return candidate;
}

function resolveAsset(pathname) {
  if (pathname === "/") return resolve(serveRoot, "index.html");
  if (development) {
    const publicAsset = safePath(resolve(frontendRoot, "public"), pathname);
    if (publicAsset && existsSync(publicAsset) && statSync(publicAsset).isFile()) return publicAsset;
  }
  const direct = safePath(serveRoot, pathname);
  if (direct && existsSync(direct) && statSync(direct).isFile()) return direct;
  return resolve(serveRoot, "index.html");
}

const server = createServer((request, response) => {
  try {
    const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);
    const asset = resolveAsset(url.pathname);
    if (!asset || !existsSync(asset) || !statSync(asset).isFile()) {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Not found\n");
      return;
    }
    response.writeHead(200, {
      "Content-Type": contentTypes.get(extname(asset)) || "application/octet-stream",
      "Cache-Control": development ? "no-store" : "public, max-age=300",
      "X-Content-Type-Options": "nosniff",
    });
    createReadStream(asset).pipe(response);
  } catch (error) {
    response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    response.end(`Server error: ${error.message}\n`);
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`LLM ethics dashboard: http://localhost:${port}`);
  console.log(`Serving: ${serveRoot}`);
});
