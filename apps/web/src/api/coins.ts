import { api } from "./client";

export interface CoinTransaction {
  id: string;
  userId: string;
  amount: number;
  reason: string;
  relatedArtworkId: string | null;
  chainTxHash: string | null;
  balanceAfter: number;
  createdAt: string;
}

export interface CoinBalance {
  balance: number;
  recentTransactions: CoinTransaction[];
}

export const getCoinBalance = () => api.get<CoinBalance>("/coins/balance");
