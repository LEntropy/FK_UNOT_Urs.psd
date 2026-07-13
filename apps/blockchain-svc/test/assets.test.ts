import { keccak256, toUtf8Bytes, Wallet } from "ethers";
import request from "supertest";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { startTestChain } from "./anvil.js";

// env.ts validates process.env at import time, so every module that pulls it
// in (contract.ts -> app.ts) must be imported dynamically, after we've set
// the env vars to point at our local anvil chain below.
let app: import("../src/app.js")["createApp"] extends (...args: infer _A) => infer R ? R : never;
let stopChain: () => Promise<void>;
let ownerAddress: string;

beforeAll(async () => {
  const chain = await startTestChain();
  stopChain = chain.stop;
  ownerAddress = new Wallet(chain.relayerPrivateKey).address;

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

describe("GET /health", () => {
  it("returns ok", async () => {
    const res = await request(app).get("/health");
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: "ok" });
  });
});

describe("POST /assets/register + GET /assets/verify", () => {
  it("registers a new artwork and verify reflects it", async () => {
    const registerRes = await request(app)
      .post("/assets/register")
      .send({ ownerAddress, content: "artwork-alpha", doNotTrain: true });

    expect(registerRes.status).toBe(201);
    expect(registerRes.body.ownerAddress).toBe(ownerAddress);
    expect(registerRes.body.doNotTrain).toBe(true);
    expect(registerRes.body.txHash).toMatch(/^0x[0-9a-f]{64}$/);

    const expectedHash = keccak256(toUtf8Bytes("artwork-alpha"));
    expect(registerRes.body.contentHash).toBe(expectedHash);

    const verifyRes = await request(app).get(`/assets/verify/${expectedHash}`);
    expect(verifyRes.status).toBe(200);
    expect(verifyRes.body).toMatchObject({
      contentHash: expectedHash,
      exists: true,
      owner: ownerAddress,
      doNotTrain: true,
    });
    expect(verifyRes.body.timestamp).toEqual(expect.any(Number));
  });

  it("computes contentHash from perceptualHash + metadataHash when both are given", async () => {
    const perceptualHash = keccak256(toUtf8Bytes("phash-artwork-beta"));
    const metadataHash = keccak256(toUtf8Bytes("meta-artwork-beta"));

    const res = await request(app)
      .post("/assets/register")
      .send({ ownerAddress, perceptualHash, metadataHash, doNotTrain: false });

    expect(res.status).toBe(201);
    // Same formula as computeContentHash() in src/hash.ts.
    const expectedHash = keccak256(perceptualHash + metadataHash.slice(2));
    expect(res.body.contentHash).toBe(expectedHash);
  });

  it("rejects a duplicate content hash with 409", async () => {
    await request(app).post("/assets/register").send({ ownerAddress, content: "artwork-gamma", doNotTrain: false });

    const dup = await request(app)
      .post("/assets/register")
      .send({ ownerAddress, content: "artwork-gamma", doNotTrain: false });

    expect(dup.status).toBe(409);
    expect(dup.body.error).toMatch(/already registered/i);
  });

  it("returns exists: false for an unregistered hash", async () => {
    const res = await request(app).get("/assets/verify/never-registered-artwork");
    expect(res.status).toBe(200);
    expect(res.body.exists).toBe(false);
    expect(res.body.owner).toBeNull();
  });

  it("rejects register requests missing required fields", async () => {
    const res = await request(app).post("/assets/register").send({ ownerAddress });
    expect(res.status).toBe(400);
  });

  it("rejects register requests with an invalid owner address", async () => {
    const res = await request(app)
      .post("/assets/register")
      .send({ ownerAddress: "not-an-address", content: "artwork-delta" });
    expect(res.status).toBe(400);
  });
});
