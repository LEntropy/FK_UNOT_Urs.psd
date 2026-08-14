import { randomUUID } from "node:crypto";
import { eq } from "drizzle-orm";
import type { Db } from "./db/client.js";
import { coinBalances, coinTransactions } from "./db/schema.js";

/**
 * Coin ledger (2026-08-10). No real top-up path exists yet -- the only way
 * a balance ever grows is the signup bonus, lazily granted the first time
 * a user touches any coin-spending action (this service has no "account
 * creation" event of its own to hook, see schema.ts's module doc).
 */

export const SIGNUP_BONUS_COINS = 100;

export const COIN_COSTS = {
  STRONG_PROTECTION: 1,
  ORIGINAL_PREVIEW_UNLOCK: 1,
  LORA_GENERATION: 1,
} as const;

export class InsufficientCoinsError extends Error {
  constructor(
    public required: number,
    public balance: number,
  ) {
    super(`insufficient coins: need ${required}, have ${balance}`);
    this.name = "InsufficientCoinsError";
  }
}

export function getOrCreateBalance(db: Db, userId: string): number {
  const existing = db.select().from(coinBalances).where(eq(coinBalances.userId, userId)).get();
  if (existing) return existing.balance;

  const now = new Date();
  db.insert(coinBalances).values({ userId, balance: SIGNUP_BONUS_COINS, updatedAt: now }).run();
  db.insert(coinTransactions)
    .values({
      id: `cnt_${randomUUID().replace(/-/g, "").slice(0, 16)}`,
      userId,
      amount: SIGNUP_BONUS_COINS,
      reason: "signup_bonus",
      relatedArtworkId: null,
      chainTxHash: null,
      balanceAfter: SIGNUP_BONUS_COINS,
      createdAt: now,
    })
    .run();
  return SIGNUP_BONUS_COINS;
}

/** Throws InsufficientCoinsError if the user can't afford `amount`. Caller
 * is expected to run this inside a db transaction alongside whatever the
 * spend is actually paying for, so a downstream failure rolls the spend
 * back too. */
export function spendCoins(
  db: Db,
  userId: string,
  amount: number,
  reason: string,
  relatedArtworkId?: string,
): void {
  const balance = getOrCreateBalance(db, userId);
  if (balance < amount) throw new InsufficientCoinsError(amount, balance);

  const balanceAfter = balance - amount;
  const now = new Date();
  db.update(coinBalances).set({ balance: balanceAfter, updatedAt: now }).where(eq(coinBalances.userId, userId)).run();
  db.insert(coinTransactions)
    .values({
      id: `cnt_${randomUUID().replace(/-/g, "").slice(0, 16)}`,
      userId,
      amount: -amount,
      reason,
      relatedArtworkId: relatedArtworkId ?? null,
      chainTxHash: null,
      balanceAfter,
      createdAt: now,
    })
    .run();
}
