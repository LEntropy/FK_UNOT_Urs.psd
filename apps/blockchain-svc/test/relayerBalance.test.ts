import request from "supertest";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { startTestChain } from "./anvil.js";

let app: import("../src/app.js")["createApp"] extends (...args: infer _A) => infer R ? R : never;
let stopChain: () => Promise<void>;

beforeAll(async () => {
  const chain = await startTestChain();
  stopChain = chain.stop;

  process.env.AMOY_RPC_URL = chain.rpcUrl;
  process.env.RELAYER_PRIVATE_KEY = chain.relayerPrivateKey;
  process.env.REGISTRY_ADDRESS = chain.registryAddress;
  process.env.PORT = "0";

  const { createApp } = await import("../src/app.js");
  app = createApp();
});

afterAll(async () => {
  await stopChain();
});

describe("GET /relayer/balance", () => {
  it("reports the real on-chain balance of the relayer wallet, not a mock", async () => {
    const res = await request(app).get("/relayer/balance");
    expect(res.status).toBe(200);
    expect(res.body.address).toMatch(/^0x[0-9a-fA-F]{40}$/);
    // anvil's default deployer account starts with a large pre-funded
    // balance (10000 ETH) -- comfortably above any sane threshold, so this
    // also proves lowBalance correctly reads false when funds are healthy.
    expect(Number(res.body.balanceEther)).toBeGreaterThan(1);
    expect(res.body.lowBalance).toBe(false);
  });

  it("flags lowBalance when the configured threshold is set above the real balance", async () => {
    // Same real chain, same real relayer wallet -- only the threshold
    // changes, proving the comparison itself (not just the RPC call).
    const { getRelayerBalance } = await import("../src/relayerBalance.js");

    const balance = await getRelayerBalance("999999");
    expect(balance.lowBalance).toBe(true);
    expect(balance.thresholdEther).toBe("999999");
  });
});
