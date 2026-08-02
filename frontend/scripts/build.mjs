import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { syncDashboardData } from "./sync-data.mjs";

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const destination = resolve(frontendRoot, "dist");

await syncDashboardData();
await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
await cp(resolve(frontendRoot, "index.html"), resolve(destination, "index.html"));
await cp(resolve(frontendRoot, "src"), resolve(destination, "src"), { recursive: true });
await cp(resolve(frontendRoot, "public"), destination, { recursive: true });

console.log(`Built static dashboard: ${destination}`);
