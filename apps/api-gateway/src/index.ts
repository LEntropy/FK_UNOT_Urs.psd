import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { createApp } from "./app.js";
import { createDb } from "./db/client.js";
import { env } from "./env.js";

mkdirSync(dirname(env.DATABASE_URL), { recursive: true });
const db = createDb(env.DATABASE_URL);

createApp(db).listen(env.PORT, () => {
  console.log(`api-gateway listening on http://localhost:${env.PORT}`);
});
