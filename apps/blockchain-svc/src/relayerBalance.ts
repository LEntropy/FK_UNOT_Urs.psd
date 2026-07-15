import { formatEther } from "ethers";
import { provider, relayerWallet } from "./contract.js";
import { env } from "./env.js";

/**
 * The relayer wallet running low on funds has caused real, silent
 * on-chain-registration failures this project has already hit in
 * practice (a "your gas ran out" error only visible after the fact, in
 * blockchain-svc's own logs, unless someone happened to be watching).
 * This makes that state queryable (GET /relayer/balance) and pollable
 * (startBalancePoller) instead of only discoverable after a registration
 * already failed.
 */
export interface RelayerBalance {
  address: string;
  balanceWei: string;
  balanceEther: string;
  thresholdEther: string;
  lowBalance: boolean;
}

// thresholdEther defaults to the env-configured value but is overridable
// (tests use this -- env.ts parses process.env once at import time, so
// reassigning process.env after the fact has no effect on it).
export async function getRelayerBalance(
  thresholdEther: string = env.RELAYER_LOW_BALANCE_THRESHOLD_ETHER,
): Promise<RelayerBalance> {
  const balanceWei = await provider.getBalance(relayerWallet.address);
  const balanceEther = formatEther(balanceWei);
  return {
    address: relayerWallet.address,
    balanceWei: balanceWei.toString(),
    balanceEther,
    thresholdEther,
    lowBalance: Number(balanceEther) < Number(thresholdEther),
  };
}

/**
 * No email/Slack/PagerDuty integration exists anywhere in this project --
 * a console warning is the honest, real extent of "alerting" this PoC can
 * do without inventing a notification channel nobody asked for. Still a
 * real improvement over the status quo (finding out only when a
 * registration fails), since this at least lands in the same process
 * logs an operator would already be watching (matching how the Pi
 * deployment's *.log files are the actual monitoring surface today).
 */
export function startBalancePoller(intervalMs: number): NodeJS.Timeout {
  const check = async () => {
    try {
      const balance = await getRelayerBalance();
      if (balance.lowBalance) {
        console.warn(
          `[relayer-balance] LOW: ${balance.address} has ${balance.balanceEther} (threshold ${balance.thresholdEther}) -- on-chain registrations will start failing once this reaches 0`,
        );
      }
    } catch (err) {
      console.warn(`[relayer-balance] check failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  void check(); // check once immediately, not just after the first interval
  return setInterval(check, intervalMs);
}
