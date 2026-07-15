import "dotenv/config";
import { z } from "zod";

const envSchema = z.object({
  DATABASE_URL: z.string().default("./data/api-gateway.db"),
  PORT: z.coerce.number().default(4000),
  ASSET_SERVICE_URL: z.string().url().default("http://localhost:3002"),
  // Delivery Gateway (apps/delivery-gateway): this service is the only
  // caller of its /internal/sign endpoint -- see that service's README's
  // trust-boundary note. The browser then hits the returned signed URL on
  // delivery-gateway directly (image bytes don't need to round-trip
  // through this gateway).
  DELIVERY_GATEWAY_URL: z.string().url().default("http://localhost:4500"),

  // HS256 shared-secret JWT for now -- the real KMS server only implements
  // envelope-key decrypt (no Sign()), so KMS-backed JWT signing isn't
  // possible without a C-side change. See src/auth/jwt.ts for the swap-out
  // point once that's added.
  JWT_ACCESS_SECRET: z.string().default("dev-insecure-access-secret-change-me"),
  JWT_REFRESH_SECRET: z.string().default("dev-insecure-refresh-secret-change-me"),
  JWT_ACCESS_TTL: z.string().default("15m"),
  JWT_REFRESH_TTL: z.string().default("30d"),

  // Custodial-wallet envelope encryption (PROJECT_DESIGN.md §3-1, §6):
  // signup generates a real wallet, the private key is RSA-wrapped for this
  // org/key and only the ciphertext is stored -- see src/auth/wallet.ts.
  KMS_HOST: z.string().default("Philosophyz.iptime.org"),
  KMS_PORT: z.coerce.number().default(8443),
  KMS_CA_CERT_PATH: z.string().default("./kms-keys/kms_ca.crt"),
  KMS_PUBLIC_KEY_PATH: z.string().default("./kms-keys/teamA1_key_v1_pub.pem"),
  KMS_ORG: z.string().default("teamA/teamA1"),
  KMS_KEY_ID: z.string().default("key_v1"),

  // Social login (src/auth/oauth.ts). Unset by default -- routes/oauth.ts
  // returns 501 for a provider whose credentials aren't configured rather
  // than a confusing failure. Get real values from Google Cloud Console
  // (APIs & Services > Credentials > OAuth client ID) / Kakao Developers
  // (내 애플리케이션 > 앱 키 + 카카오 로그인 활성화) -- only the project
  // owner can create these, not something this code can supply.
  GOOGLE_CLIENT_ID: z.string().optional(),
  GOOGLE_CLIENT_SECRET: z.string().optional(),
  KAKAO_CLIENT_ID: z.string().optional(),
  KAKAO_CLIENT_SECRET: z.string().optional(),
  // Must match each provider's registered redirect URI exactly
  // (PUBLIC_URL + /auth/google/callback or /auth/kakao/callback).
  PUBLIC_URL: z.string().url().default("http://localhost:4000"),
  // Where to send the browser after a successful OAuth login, with the
  // token pair attached -- apps/web's OAuthCallbackPage reads it from there.
  WEB_URL: z.string().url().default("http://localhost:5173"),
});

export const env = envSchema.parse(process.env);
