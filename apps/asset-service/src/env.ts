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

  // Original-image envelope encryption at rest (src/crypto/imageEncryption.ts).
  // wrapKey() (encrypt path, upload time) is client-side only -- no live KMS
  // server needed, so KMS_PUBLIC_KEY_PATH is the only one of these required
  // just to accept an upload. The rest are only touched by decryptToTempFile()
  // right before protection-svc needs a real file (orchestration.ts).
  KMS_PUBLIC_KEY_PATH: z.string().default("./kms-keys/teamA1_key_v1_pub.pem"),
  KMS_HOST: z.string().default("127.0.0.1"),
  KMS_PORT: z.coerce.number().default(8443),
  KMS_CA_CERT_PATH: z.string().default("./kms-keys/kms_ca.crt"),
  KMS_ORG: z.string().default("teamA/teamA1"),
  KMS_KEY_ID: z.string().default("key_v1"),
});

export const env = envSchema.parse(process.env);
