// Minimal static file server for the production build (dist/). The Pi has
// no nginx and every other service here is deployed as a native process
// (see deploy/pi/), so this matches that pattern instead of introducing a
// new one just for static files.
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const port = process.env.PORT ?? 5173;

const app = express();
app.use(express.static(path.join(__dirname, "dist")));
app.get("*", (_req, res) => res.sendFile(path.join(__dirname, "dist", "index.html")));

app.listen(port, () => console.log(`web listening on http://localhost:${port}`));
