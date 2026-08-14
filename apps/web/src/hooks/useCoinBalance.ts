import { useCallback, useEffect, useState } from "react";
import { useAuthStore } from "../store/auth";
import { getCoinBalance } from "../api/coins";

/** Fetches the logged-in user's coin balance on mount/login, with a
 * `refresh()` callers use right after a spend (upload, unlock, LoRA job)
 * so the displayed number doesn't lag a full page reload behind. */
export function useCoinBalance() {
  const user = useAuthStore((s) => s.user);
  const [balance, setBalance] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    if (!user) {
      setBalance(null);
      return;
    }
    try {
      const res = await getCoinBalance();
      setBalance(res.balance);
    } catch {
      // Best-effort -- a failed balance fetch shouldn't block the rest of
      // the page; whatever triggered a spend will surface its own error.
    }
  }, [user]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { balance, refresh };
}
