import { migrate } from "drizzle-orm/better-sqlite3/migrator";
import { createDb } from "./client.js";
import { env } from "../env.js";

const db = createDb(env.DATABASE_URL);
migrate(db, { migrationsFolder: "./drizzle" });
console.log(`migrated ${env.DATABASE_URL}`);
