import { env } from "../env.js";

/**
 * Google and Kakao OAuth2 authorization-code flow, implemented directly
 * against each provider's REST endpoints (no OAuth client library --
 * both flows are a handful of fetch calls, not worth a dependency).
 *
 * Neither provider is configured by default (env.ts's *_CLIENT_ID/SECRET
 * are optional) -- routes/oauth.ts returns a clear 501 rather than a
 * confusing failure when a provider's credentials aren't set. Getting
 * real credentials means registering an app in each provider's developer
 * console (Google Cloud Console / Kakao Developers) -- something only the
 * project owner can do, not something this code can supply.
 */

export interface OAuthProfile {
  providerUserId: string;
  email: string;
  displayName: string | null;
  avatarUri: string | null;
}

export function isProviderConfigured(provider: "GOOGLE" | "KAKAO"): boolean {
  if (provider === "GOOGLE") return Boolean(env.GOOGLE_CLIENT_ID && env.GOOGLE_CLIENT_SECRET);
  return Boolean(env.KAKAO_CLIENT_ID && env.KAKAO_CLIENT_SECRET);
}

export function buildAuthorizeUrl(provider: "GOOGLE" | "KAKAO", state: string): string {
  if (provider === "GOOGLE") {
    const params = new URLSearchParams({
      client_id: env.GOOGLE_CLIENT_ID!,
      redirect_uri: `${env.PUBLIC_URL}/auth/google/callback`,
      response_type: "code",
      scope: "openid email profile",
      state,
    });
    return `https://accounts.google.com/o/oauth2/v2/auth?${params}`;
  }

  const params = new URLSearchParams({
    client_id: env.KAKAO_CLIENT_ID!,
    redirect_uri: `${env.PUBLIC_URL}/auth/kakao/callback`,
    response_type: "code",
    state,
  });
  return `https://kauth.kakao.com/oauth/authorize?${params}`;
}

export async function exchangeCodeForProfile(provider: "GOOGLE" | "KAKAO", code: string): Promise<OAuthProfile> {
  return provider === "GOOGLE" ? exchangeGoogle(code) : exchangeKakao(code);
}

async function exchangeGoogle(code: string): Promise<OAuthProfile> {
  const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: env.GOOGLE_CLIENT_ID!,
      client_secret: env.GOOGLE_CLIENT_SECRET!,
      code,
      grant_type: "authorization_code",
      redirect_uri: `${env.PUBLIC_URL}/auth/google/callback`,
    }),
  });
  if (!tokenRes.ok) {
    throw new Error(`Google token exchange failed: ${tokenRes.status} ${await tokenRes.text()}`);
  }
  const { access_token } = (await tokenRes.json()) as { access_token: string };

  const profileRes = await fetch("https://www.googleapis.com/oauth2/v2/userinfo", {
    headers: { Authorization: `Bearer ${access_token}` },
  });
  if (!profileRes.ok) {
    throw new Error(`Google userinfo fetch failed: ${profileRes.status} ${await profileRes.text()}`);
  }
  const profile = (await profileRes.json()) as { id: string; email: string; name?: string; picture?: string };

  return {
    providerUserId: profile.id,
    email: profile.email,
    displayName: profile.name ?? null,
    avatarUri: profile.picture ?? null,
  };
}

async function exchangeKakao(code: string): Promise<OAuthProfile> {
  const tokenRes = await fetch("https://kauth.kakao.com/oauth/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      client_id: env.KAKAO_CLIENT_ID!,
      client_secret: env.KAKAO_CLIENT_SECRET!,
      code,
      redirect_uri: `${env.PUBLIC_URL}/auth/kakao/callback`,
    }),
  });
  if (!tokenRes.ok) {
    throw new Error(`Kakao token exchange failed: ${tokenRes.status} ${await tokenRes.text()}`);
  }
  const { access_token } = (await tokenRes.json()) as { access_token: string };

  const profileRes = await fetch("https://kapi.kakao.com/v2/user/me", {
    headers: { Authorization: `Bearer ${access_token}` },
  });
  if (!profileRes.ok) {
    throw new Error(`Kakao userinfo fetch failed: ${profileRes.status} ${await profileRes.text()}`);
  }
  const profile = (await profileRes.json()) as {
    id: number;
    kakao_account?: { email?: string; profile?: { nickname?: string; profile_image_url?: string } };
  };

  const email = profile.kakao_account?.email;
  if (!email) {
    // Kakao only returns email if the app has that consent item approved
    // AND the user granted it -- a real, common failure mode, not an edge
    // case to silently work around with a fake placeholder email.
    throw new Error(
      "Kakao did not return an email for this account -- check the app's consent items in Kakao Developers, and that the user granted email access",
    );
  }

  return {
    providerUserId: String(profile.id),
    email,
    displayName: profile.kakao_account?.profile?.nickname ?? null,
    avatarUri: profile.kakao_account?.profile?.profile_image_url ?? null,
  };
}
