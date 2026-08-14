import { createApp } from "./app.js";
import { env } from "./env.js";
import { startBalancePoller } from "./relayerBalance.js";

// ethers' JsonRpcProvider does its own background network-detection polling
// outside any call this service explicitly awaits (see contract.ts's
// provider) -- a transient RPC blip there rejects a promise with no
// catch site of ours attached, which Node treats as fatal by default and
// kills the whole process. Found live: a relayer RPC ETIMEDOUT took down
// blockchain-svc mid-registration, which is what actually produced a real
// artwork's "fetch failed" on asset-service's side, not any bug in the
// registration call itself. Logging instead of crashing is strictly safer
// here -- this service has no per-request state that a stray rejection
// could leave corrupted.
process.on("unhandledRejection", (reason) => {
  console.warn(`[unhandled-rejection] ${reason instanceof Error ? reason.stack ?? reason.message : String(reason)}`);
});

createApp().listen(env.PORT, () => {
  console.log(`blockchain-svc listening on http://localhost:${env.PORT}`);
});

startBalancePoller(env.RELAYER_BALANCE_POLL_INTERVAL_SECONDS * 1000);
