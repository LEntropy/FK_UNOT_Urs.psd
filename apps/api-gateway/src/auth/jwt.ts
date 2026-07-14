import jwt from "jsonwebtoken";
import { env } from "../env.js";

/**
 * Single swap-out point for JWT signing. HS256 + shared secret today
 * because the live KMS server only implements envelope-key decrypt (no
 * Sign() RPC) -- see PROJECT_DESIGN.md §6 vs. the actual protocol.c.
 * Once KMS grows a signing operation, only this file needs to change.
 */

export interface AccessTokenPayload {
  sub: string; // user id
  role: string;
  walletAddress: string;
}

export function signAccessToken(payload: AccessTokenPayload): string {
  return jwt.sign(payload, env.JWT_ACCESS_SECRET, { expiresIn: env.JWT_ACCESS_TTL } as jwt.SignOptions);
}

export function signRefreshToken(userId: string): string {
  return jwt.sign({ sub: userId, type: "refresh" }, env.JWT_REFRESH_SECRET, {
    expiresIn: env.JWT_REFRESH_TTL,
  } as jwt.SignOptions);
}

export function verifyAccessToken(token: string): AccessTokenPayload {
  return jwt.verify(token, env.JWT_ACCESS_SECRET) as AccessTokenPayload;
}

export function verifyRefreshToken(token: string): { sub: string } {
  const decoded = jwt.verify(token, env.JWT_REFRESH_SECRET) as { sub: string; type: string };
  if (decoded.type !== "refresh") {
    throw new Error("not a refresh token");
  }
  return decoded;
}
