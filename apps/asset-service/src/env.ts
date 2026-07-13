import "dotenv/config";
import { z } from "zod";

const envSchema = z.object({
  DATABASE_URL: z.string().default("./data/asset-service.db"),
  PROTECTION_SVC_URL: z.string().url().default("http://localhost:8000"),
  BLOCKCHAIN_SVC_URL: z.string().url().default("http://localhost:3001"),
  PORT: z.coerce.number().default(3002),
  // For display/bookkeeping in ownership_records only -- blockchain-svc is
  // the source of truth for which contract it actually registered against.
  // Defaults to the currently deployed Amoy testnet contract, see
  // contracts/DEPLOYMENTS.md.
  CHAIN_NAME: z.string().default("polygon-amoy"),
  REGISTRY_ADDRESS: z.string().default("0x12fe026abacd896956ccf71044640af04c7e8a97"),
});

export const env = envSchema.parse(process.env);
