import "dotenv/config";
import { z } from "zod";

const envSchema = z.object({
  AMOY_RPC_URL: z.string().url(),
  RELAYER_PRIVATE_KEY: z.string().regex(/^0x[0-9a-fA-F]{64}$/, "must be a 32-byte hex private key"),
  REGISTRY_ADDRESS: z.string().regex(/^0x[0-9a-fA-F]{40}$/, "must be a 20-byte hex address"),
  PORT: z.coerce.number().default(3001),
});

export const env = envSchema.parse(process.env);
