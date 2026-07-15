import { randomBytes, randomUUID } from "node:crypto";
import { Router } from "express";
import { and, eq } from "drizzle-orm";
import type { Db } from "../db/client.js";
import { users } from "../db/schema.js";
import { signAccessToken, signRefreshToken } from "../auth/jwt.js";
import { provisionCustodialWallet } from "../auth/wallet.js";
import { isProviderConfigured, buildAuthorizeUrl, exchangeCodeForProfile, type OAuthProfile } from "../auth/oauth.js";
import { env } from "../env.js";

// CSRF state tokens: short-lived, in-memory (this is a single-process
// service, no need for Redis/DB for a 10-minute nonce). A provider that
// restarts the process between /auth/google and its callback would fail
// the login and need a retry -- an acceptable trade-off, not silently
// skipping CSRF protection to avoid it.
const pendingStates = new Map<string, number>();
const STATE_TTL_MS = 10 * 60 * 1000;

function issueState(): string {
  const state = randomBytes(16).toString("hex");
  pendingStates.set(state, Date.now() + STATE_TTL_MS);
  return state;
}

function consumeState(state: string | undefined): boolean {
  if (!state) return false;
  const expiry = pendingStates.get(state);
  pendingStates.delete(state);
  return expiry !== undefined && expiry > Date.now();
}

export function oauthRouter(db: Db): Router {
  const router = Router();

  for (const provider of ["google", "kakao"] as const) {
    const PROVIDER = provider.toUpperCase() as "GOOGLE" | "KAKAO";

    router.get(`/${provider}`, (_req, res) => {
      if (!isProviderConfigured(PROVIDER)) {
        return res
          .status(501)
          .json({ error: `${PROVIDER} login is not configured (missing ${PROVIDER}_CLIENT_ID/SECRET)` });
      }
      res.redirect(buildAuthorizeUrl(PROVIDER, issueState()));
    });

    router.get(`/${provider}/callback`, async (req, res) => {
      if (!isProviderConfigured(PROVIDER)) {
        return res.status(501).json({ error: `${PROVIDER} login is not configured` });
      }
      if (!consumeState(req.query.state as string | undefined)) {
        return res.status(400).json({ error: "invalid or expired OAuth state" });
      }
      const code = req.query.code as string | undefined;
      if (!code) {
        return res.status(400).json({ error: "missing code" });
      }

      let profile: OAuthProfile;
      try {
        profile = await exchangeCodeForProfile(PROVIDER, code);
      } catch (err) {
        return res.status(502).json({ error: err instanceof Error ? err.message : String(err) });
      }

      const user = findOrCreateOAuthUser(db, PROVIDER, profile);

      const accessToken = signAccessToken({ sub: user.id, role: user.role, walletAddress: user.walletAddress });
      const refreshToken = signRefreshToken(user.id);

      // Fragment, not query string -- tokens shouldn't end up in server
      // access logs or a Referer header if the callback page links out.
      const redirect = new URL("/oauth-callback", env.WEB_URL);
      redirect.hash = new URLSearchParams({ accessToken, refreshToken }).toString();
      res.redirect(redirect.toString());
    });
  }

  return router;
}

function findOrCreateOAuthUser(db: Db, provider: "GOOGLE" | "KAKAO", profile: OAuthProfile) {
  const existing = db
    .select()
    .from(users)
    .where(and(eq(users.authProvider, provider), eq(users.providerUserId, profile.providerUserId)))
    .get();
  if (existing) return existing;

  // Same email under a different provider (or LOCAL) is a separate account
  // here -- no auto-merge (see schema.ts's doc comment on why). handle must
  // be unique platform-wide, so a provider-derived one needs a random
  // suffix on collision rather than failing signup outright.
  const baseHandle = (profile.displayName ?? profile.email.split("@")[0]).replace(/[^a-zA-Z0-9_]/g, "_").slice(0, 24) || "user";
  let handle = baseHandle;
  while (db.select().from(users).where(eq(users.handle, handle)).get()) {
    handle = `${baseHandle}_${randomBytes(2).toString("hex")}`;
  }

  const wallet = provisionCustodialWallet();
  const id = `usr_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
  const row = {
    id,
    email: profile.email,
    passwordHash: null,
    authProvider: provider,
    providerUserId: profile.providerUserId,
    handle,
    displayName: profile.displayName ?? handle,
    avatarUri: profile.avatarUri,
    role: "CREATOR" as const,
    walletAddress: wallet.address,
    encryptedWalletKey: wallet.encryptedPrivateKeyBase64,
    createdAt: new Date(),
    status: "ACTIVE" as const,
  };
  db.insert(users).values(row).run();
  return row;
}
