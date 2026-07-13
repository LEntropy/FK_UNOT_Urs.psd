import { type ChildProcess, spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { ContractFactory, JsonRpcProvider, Wallet } from "ethers";

const ARTIFACT_PATH = fileURLToPath(
  new URL("../../../contracts/out/OwnershipRegistry.sol/OwnershipRegistry.json", import.meta.url),
);

/**
 * Spawns a local anvil node on a random free port, deploys OwnershipRegistry
 * to it from the forge-compiled artifact, and authorizes the deployer as a
 * relayer — the same setup we did manually against Amoy testnet, but
 * ephemeral and free. Requires `contracts/` to have been built
 * (`forge build`) at least once.
 */
export async function startTestChain() {
  const anvil = await spawnAnvil();

  try {
    // cacheTimeout: -1 disables ethers' short-lived read cache. Anvil
    // auto-mines each tx instantly, faster than ethers' default polling
    // interval notices — without this, the second of two back-to-back txs
    // (deploy then setRelayer) reuses a stale cached nonce and gets rejected
    // as "nonce too low".
    const provider = new JsonRpcProvider(anvil.rpcUrl, undefined, { cacheTimeout: -1 });
    const deployerWallet = new Wallet(anvil.deployerKey, provider);

    const artifact = JSON.parse(readFileSync(ARTIFACT_PATH, "utf-8"));
    const factory = new ContractFactory(artifact.abi, artifact.bytecode.object, deployerWallet);
    const registry = await factory.deploy(deployerWallet.address);
    await registry.waitForDeployment();

    const registryAddress = await registry.getAddress();

    // Authorize the deployer wallet as a relayer so registerFor() works,
    // mirroring the setRelayer call we made against the real testnet deployment.
    const setRelayerTx = await (registry as any).setRelayer(deployerWallet.address, true);
    await setRelayerTx.wait();

    return {
      rpcUrl: anvil.rpcUrl,
      relayerPrivateKey: anvil.deployerKey,
      registryAddress,
      async stop() {
        anvil.process.kill();
      },
    };
  } catch (err) {
    anvil.process.kill();
    throw err;
  }
}

function spawnAnvil(): Promise<{ process: ChildProcess; deployerKey: string; rpcUrl: string }> {
  // Random high port per run so leftover/orphaned processes from a previous
  // run can never collide with this one.
  const port = 20000 + Math.floor(Math.random() * 20000);

  return new Promise((resolve, reject) => {
    const anvilProcess = spawn("anvil", ["--port", String(port)], {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let output = "";
    let settled = false;

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      anvilProcess.kill();
      reject(new Error(`anvil did not start within 10s:\n${output}`));
    }, 10_000);

    const onData = (chunk: Buffer) => {
      output += chunk.toString();
      if (!settled && /Listening on 127\.0\.0\.1:\d+/.test(output)) {
        const keyMatch = output.match(/Private Keys\n=+\n+\(0\)\s+(0x[0-9a-fA-F]{64})/);
        if (!keyMatch) {
          settled = true;
          clearTimeout(timer);
          anvilProcess.kill();
          reject(new Error(`anvil started but private key could not be parsed:\n${output}`));
          return;
        }
        settled = true;
        clearTimeout(timer);
        resolve({ process: anvilProcess, deployerKey: keyMatch[1], rpcUrl: `http://127.0.0.1:${port}` });
      }
    };

    anvilProcess.stdout?.on("data", onData);
    anvilProcess.stderr?.on("data", (chunk: Buffer) => {
      output += chunk.toString();
    });

    anvilProcess.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`failed to spawn anvil (is it on PATH?): ${err.message}`));
    });
    anvilProcess.on("exit", (code) => {
      if (settled) return;
      if (code !== null && code !== 0) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`anvil exited early with code ${code}:\n${output}`));
      }
    });
  });
}
