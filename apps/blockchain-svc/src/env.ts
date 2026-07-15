import "dotenv/config";
import { z } from "zod";

// Relayer key: either plaintext (RELAYER_PRIVATE_KEY, local dev/tests -- see
// test/anvil.ts) or KMS-wrapped (RELAYER_ENCRYPTED_KEY, the Pi deployment --
// see src/contract.ts's resolveRelayerPrivateKey()). Exactly one is
// required; contract.ts is the single place that reads either.
const envSchema = z
  .object({
    AMOY_RPC_URL: z.string().url(),
    RELAYER_PRIVATE_KEY: z.string().regex(/^0x[0-9a-fA-F]{64}$/, "must be a 32-byte hex private key").optional(),
    RELAYER_ENCRYPTED_KEY: z.string().optional(), // base64 RSA-PKCS1 ciphertext, see infra/kms-adapter
    KMS_HOST: z.string().optional(),
    KMS_PORT: z.coerce.number().optional(),
    KMS_CA_CERT_PATH: z.string().optional(),
    KMS_ORG: z.string().optional(),
    KMS_KEY_ID: z.string().optional(),
    REGISTRY_ADDRESS: z.string().regex(/^0x[0-9a-fA-F]{40}$/, "must be a 20-byte hex address"),
    PORT: z.coerce.number().default(3001),
    // src/relayerBalance.ts. Default is a testnet-appropriate guess (Amoy
    // MATIC is free from a faucet, registrations are cheap) -- a real
    // mainnet deployment should set this based on actual observed gas
    // cost per registration, not this default.
    RELAYER_LOW_BALANCE_THRESHOLD_ETHER: z.string().default("0.05"),
    RELAYER_BALANCE_POLL_INTERVAL_SECONDS: z.coerce.number().default(300),
  })
  .refine((v) => v.RELAYER_PRIVATE_KEY || v.RELAYER_ENCRYPTED_KEY, {
    message: "one of RELAYER_PRIVATE_KEY or RELAYER_ENCRYPTED_KEY is required",
  })
  .refine(
    (v) => !v.RELAYER_ENCRYPTED_KEY || (v.KMS_HOST && v.KMS_PORT && v.KMS_CA_CERT_PATH && v.KMS_ORG && v.KMS_KEY_ID),
    { message: "RELAYER_ENCRYPTED_KEY requires KMS_HOST/KMS_PORT/KMS_CA_CERT_PATH/KMS_ORG/KMS_KEY_ID too" },
  );

export const env = envSchema.parse(process.env);
