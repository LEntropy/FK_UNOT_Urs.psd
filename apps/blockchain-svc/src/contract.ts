import { Contract, JsonRpcProvider, Wallet } from "ethers";
import { env } from "./env.js";

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
export const relayerWallet = new Wallet(env.RELAYER_PRIVATE_KEY, provider);
export const registry = new Contract(env.REGISTRY_ADDRESS, REGISTRY_ABI, relayerWallet);
