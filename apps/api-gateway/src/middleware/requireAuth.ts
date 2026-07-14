import type { NextFunction, Request, Response } from "express";
import { verifyAccessToken, type AccessTokenPayload } from "../auth/jwt.js";

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      user?: AccessTokenPayload;
    }
  }
}

export function requireAuth(req: Request, res: Response, next: NextFunction) {
  const header = req.header("authorization");
  if (!header?.startsWith("Bearer ")) {
    return res.status(401).json({ error: "missing bearer token" });
  }

  try {
    req.user = verifyAccessToken(header.slice("Bearer ".length));
    next();
  } catch {
    res.status(401).json({ error: "invalid or expired token" });
  }
}
