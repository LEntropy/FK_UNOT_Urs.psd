import { AbiCoder, keccak256, randomBytes, Wallet } from "ethers";
import { beforeAll, afterAll, describe, expect, it, vi } from "vitest";
import { startTestChain } from "./anvil.js";

// Mocked here (not a live KMS server call) -- infra/kms-adapter's own
// roundtrip.test.ts already proves unwrapKey works against the real
// server; this file only proves contract.ts wires the *result* of that
// call into the relayer Wallet correctly, and that RELAYER_PRIVATE_KEY
// is never read when RELAYER_ENCRYPTED_KEY is set (a silent fallback
// would defeat the entire point of moving the key into KMS).
const unwrapKey = vi.fn();
vi.mock("@dontai/kms-adapter", () => ({ unwrapKey }));

let stopChain: () => Promise<void>;
let relayerAddress: string;

beforeAll(async () => {
  const chain = await startTestChain();
  stopChain = chain.stop;
  relayerAddress = new Wallet(chain.relayerPrivateKey).address;

  unwrapKey.mockResolvedValue(Buffer.from(chain.relayerPrivateKey.slice(2), "hex"));

  process.env.AMOY_RPC_URL = chain.rpcUrl;
  process.env.REGISTRY_ADDRESS = chain.registryAddress;
  process.env.PORT = "0";
  delete process.env.RELAYER_PRIVATE_KEY; // prove the encrypted path doesn't need it
  process.env.RELAYER_ENCRYPTED_KEY = "ZmFrZS1jaXBoZXJ0ZXh0"; // opaque to contract.ts, only unwrapKey's mock matters
  process.env.KMS_HOST = "kms.invalid";
  process.env.KMS_PORT = "8443";
  process.env.KMS_CA_CERT_PATH = "/dev/null";
  process.env.KMS_ORG = "teamA/teamA1";
  process.env.KMS_KEY_ID = "key_v1";
});

afterAll(async () => {
  await stopChain();
});

describe("relayer key resolution via KMS", () => {
  it("calls unwrapKey with the configured org/key and never touches RELAYER_PRIVATE_KEY", async () => {
    const { relayerWallet } = await import("../src/contract.js");

    expect(unwrapKey).toHaveBeenCalledWith(
      expect.objectContaining({
        host: "kms.invalid",
        port: 8443,
        requesterOrg: "teamA/teamA1",
        fileOrg: "teamA/teamA1",
        keyId: "key_v1",
        encKey: Buffer.from("ZmFrZS1jaXBoZXJ0ZXh0", "base64"),
      }),
    );
    expect(relayerWallet.address).toBe(relayerAddress);
  });

  it("the resolved wallet can actually sign a real transaction on the test chain", async () => {
    const { provider, registry, relayerWallet } = await import("../src/contract.js");
    const contentHash = `0x${"11".repeat(32)}`;
    const doNotTrain = true;
    const nonce = `0x${Buffer.from(randomBytes(32)).toString("hex")}`;

    // Same commit-reveal formula as src/routes/register.ts's self-service
    // path (register(), not registerFor() -- this test signs directly with
    // relayerWallet as the registrant itself, not sponsoring someone else).
    const commitHash = keccak256(
      AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bool", "address", "bytes32"],
        [contentHash, doNotTrain, relayerWallet.address, nonce],
      ),
    );
    const commitTx = await registry.commit(commitHash);
    const commitReceipt = await commitTx.wait();

    const minCommitAge = Number(await registry.MIN_COMMIT_AGE());
    while ((await provider.getBlockNumber()) < commitReceipt.blockNumber + minCommitAge) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    const tx = await (registry as any).register(contentHash, doNotTrain, nonce);
    const receipt = await tx.wait();

    expect(receipt.status).toBe(1);
  });
});
