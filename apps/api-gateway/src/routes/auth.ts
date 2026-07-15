import { randomUUID } from "node:crypto";
import { Router } from "express";
import { and, eq } from "drizzle-orm";
import { z } from "zod";
import type { Db } from "../db/client.js";
import { users } from "../db/schema.js";
import { hashPassword, verifyPassword } from "../auth/password.js";
import { signAccessToken, signRefreshToken, verifyRefreshToken } from "../auth/jwt.js";
import { provisionCustodialWallet } from "../auth/wallet.js";

const signupSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  handle: z
    .string()
    .min(3)
    .max(32)
    .regex(/^[a-zA-Z0-9_]+$/, "handle must be alphanumeric/underscore"),
  displayName: z.string().optional(),
});

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
});

const refreshSchema = z.object({
  refreshToken: z.string().min(1),
});

export function authRouter(db: Db): Router {
  const router = Router();

  router.post("/signup", (req, res) => {
    const parsed = signupSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    const { email, password, handle, displayName } = parsed.data;

    // Scoped to LOCAL specifically -- (email, authProvider) is the real
    // unique constraint (schema.ts), so a Google/Kakao account with this
    // same email doesn't block a separate email/password signup.
    if (db.select().from(users).where(and(eq(users.email, email), eq(users.authProvider, "LOCAL"))).get()) {
      return res.status(409).json({ error: "email already registered" });
    }
    if (db.select().from(users).where(eq(users.handle, handle)).get()) {
      return res.status(409).json({ error: "handle already taken" });
    }

    const wallet = provisionCustodialWallet();
    const id = `usr_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
    const row = {
      id,
      email,
      passwordHash: hashPassword(password),
      authProvider: "LOCAL" as const,
      handle,
      displayName: displayName ?? handle,
      role: "CREATOR" as const,
      walletAddress: wallet.address,
      encryptedWalletKey: wallet.encryptedPrivateKeyBase64,
      createdAt: new Date(),
      status: "ACTIVE" as const,
    };
    db.insert(users).values(row).run();

    res.status(201).json({
      accessToken: signAccessToken({ sub: id, role: row.role, walletAddress: wallet.address }),
      refreshToken: signRefreshToken(id),
      user: { id, email, handle, walletAddress: wallet.address, role: row.role },
    });
  });

  router.post("/login", (req, res) => {
    const parsed = loginSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    const { email, password } = parsed.data;

    const user = db.select().from(users).where(and(eq(users.email, email), eq(users.authProvider, "LOCAL"))).get();
    // user.passwordHash is only ever null for social-login accounts, which
    // the authProvider="LOCAL" filter above already excludes -- but check
    // explicitly rather than asserting, since a null here would otherwise
    // crash verifyPassword instead of failing as a normal 401.
    if (!user || !user.passwordHash || !verifyPassword(password, user.passwordHash)) {
      return res.status(401).json({ error: "invalid email or password" });
    }

    res.json({
      accessToken: signAccessToken({ sub: user.id, role: user.role, walletAddress: user.walletAddress }),
      refreshToken: signRefreshToken(user.id),
      user: { id: user.id, email: user.email, handle: user.handle, walletAddress: user.walletAddress, role: user.role },
    });
  });

  router.post("/refresh", (req, res) => {
    const parsed = refreshSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }

    let userId: string;
    try {
      userId = verifyRefreshToken(parsed.data.refreshToken).sub;
    } catch {
      return res.status(401).json({ error: "invalid or expired refresh token" });
    }

    const user = db.select().from(users).where(eq(users.id, userId)).get();
    if (!user) {
      return res.status(401).json({ error: "user no longer exists" });
    }

    res.json({ accessToken: signAccessToken({ sub: user.id, role: user.role, walletAddress: user.walletAddress }) });
  });

  return router;
}
