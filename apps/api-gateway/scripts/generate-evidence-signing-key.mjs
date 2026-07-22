#!/usr/bin/env node
// One-off ops script: generates the Ed25519 keypair evidence bundles are
// signed with, wraps the private key for KMS_ORG/KMS_KEY_ID (same keys
// already configured for custodial wallets, see env.ts), and prints the
// base64 ciphertext to set as EVIDENCE_SIGNING_ENCRYPTED_KEY. Mirrors
// provisionCustodialWallet() in src/auth/wallet.ts -- run once per
// deployment, not part of the request path. Not run in CI/tests.
//
// Usage: node scripts/generate-evidence-signing-key.mjs [path/to/pub.pem]
// (defaults to ./kms-keys/teamA1_key_v1_pub.pem, matching .env's default
// KMS_PUBLIC_KEY_PATH)

import { generateKeyPairSync } from "node:crypto";
import { wrapKey } from "@dontai/kms-adapter";

const pubKeyPath = process.argv[2] ?? "./kms-keys/teamA1_key_v1_pub.pem";

const { privateKey, publicKey } = generateKeyPairSync("ed25519");
const privateKeyDer = privateKey.export({ type: "pkcs8", format: "der" });
const wrapped = wrapKey(pubKeyPath, privateKeyDer);

console.log("EVIDENCE_SIGNING_ENCRYPTED_KEY=" + wrapped.toString("base64"));
console.log();
console.log("Public key (for out-of-band verification, NOT a secret):");
console.log(publicKey.export({ type: "spki", format: "pem" }).toString());
