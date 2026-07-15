import { Contract, JsonRpcProvider, Wallet } from "ethers";
import { unwrapKey } from "@dontai/kms-adapter";
import { env } from "./env.js";

/**
 * KMS never signs directly (the real C KMS server only implements
 * envelope-key decrypt, no Sign() RPC -- see infra/kms-adapter's own
 * docs) -- it holds the relayer's private key encrypted at rest and this
 * decrypts it once at startup so plaintext never sits in the deployed
 * .env file. Local dev/tests use the plaintext RELAYER_PRIVATE_KEY path
 * instead (env.ts's refine() requires exactly the right combination of
 * vars for whichever path is active).
 */
async function resolveRelayerPrivateKey(): Promise<string> {
  if (!env.RELAYER_ENCRYPTED_KEY) {
    return env.RELAYER_PRIVATE_KEY!;
  }

  const plainKey = await unwrapKey({
    host: env.KMS_HOST!,
    port: env.KMS_PORT!,
    caCertPath: env.KMS_CA_CERT_PATH!,
    requesterOrg: env.KMS_ORG!,
    fileOrg: env.KMS_ORG!,
    keyId: env.KMS_KEY_ID!,
    encKey: Buffer.from(env.RELAYER_ENCRYPTED_KEY, "base64"),
  });
  return `0x${plainKey.toString("hex")}`;
}

// Human-readable ABI — kept in sync manually with contracts/src/OwnershipRegistry.sol.
// Only the functions/events this service actually calls are listed.
const REGISTRY_ABI = [
  "function register(bytes32 contentHash, bool doNotTrain) returns (uint256 id)",
  "function registerFor(address owner, bytes32 contentHash, bool doNotTrain) returns (uint256 id)",
  "function verify(bytes32 contentHash) view returns (bool exists, address owner, uint64 timestamp, bool doNotTrain)",
  "function relayers(address) view returns (bool)",
  "event Registered(uint256 indexed id, address indexed owner, bytes32 contentHash, bool doNotTrain)",
  "error AlreadyRegistered(bytes32 contentHash)",
  "error NotFound(uint256 id)",
  "error NotOwner(uint256 id, address caller)",
  "error NotRelayer(address caller)",
  "error ZeroAddress()",
];

// cacheTimeout: -1 disables ethers' short-lived read cache (default ~250ms).
// Without it, back-to-back registrations from the same relayer wallet can
// read a stale nonce and fail with "nonce too low" on fast-mining chains —
// see test/anvil.ts for the full explanation. Cheap to disable in general;
// note this does not by itself solve nonce races between *concurrent*
// requests, which would need a queued/serialized sender if volume grows.
export const provider = new JsonRpcProvider(env.AMOY_RPC_URL, undefined, { cacheTimeout: -1 });
export const relayerWallet = new Wallet(await resolveRelayerPrivateKey(), provider);
export const registry = new Contract(env.REGISTRY_ADDRESS, REGISTRY_ABI, relayerWallet);
