pub mod crawlers;
pub mod rate_limit;
pub mod signing;

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rate_limit::RateLimiter;
use serde::{Deserialize, Serialize};

pub struct AppState {
    pub signing_secret: String,
    pub asset_service_url: String,
    pub allowed_referer_hosts: Vec<String>,
    pub http: reqwest::Client,
    pub rate_limiter: RateLimiter,
    pub sign_ttl_seconds: u64,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/health",
            get(|| async { Json(serde_json::json!({"status": "ok"})) }),
        )
        .route("/robots.txt", get(robots_txt))
        .route("/internal/sign", post(sign_url))
        .route("/asset/{id}/render", get(render_asset))
        .with_state(state)
}

async fn robots_txt() -> impl IntoResponse {
    let mut body = String::from(
        "User-agent: *\nDisallow: /asset/\n\n\
         # PROJECT_DESIGN.md \u{00a7}3-5: this file is a cooperative signal only,\n\
         # not enforcement -- real access control happens in /asset/:id/render's\n\
         # own crawler classification (see src/crawlers.rs), which a well-behaved\n\
         # crawler that ignores this file still can't bypass.\n",
    );
    for ua in crawlers::AI_CRAWLER_USER_AGENTS {
        body.push_str(&format!("\nUser-agent: {ua}\nDisallow: /\n"));
    }
    ([("content-type", "text/plain; charset=utf-8")], body)
}

#[derive(Deserialize)]
struct SignRequest {
    #[serde(rename = "artworkId")]
    artwork_id: String,
    viewer: Viewer,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Viewer {
    Anonymous,
    LoggedIn,
    Thumbnail,
}

impl Viewer {
    /// PROJECT_DESIGN.md \u{00a7}3-5: "비로그인 → 1280px, 로그인 유저 → 2048px".
    /// `Thumbnail` isn't in the original design text but reuses rust-core's
    /// already-built grid_thumbnail_512 variant for gallery views -- not a
    /// new capability, just exposing an existing variant through this gate
    /// too instead of leaving it unreachable.
    fn variant(self) -> &'static str {
        match self {
            Viewer::Anonymous => "public_preview_1280",
            Viewer::LoggedIn => "public_preview_2048",
            Viewer::Thumbnail => "grid_thumbnail_512",
        }
    }
}

#[derive(Serialize)]
struct SignResponse {
    url: String,
}

/// Issues a signed, short-TTL render URL. Same trust boundary as every
/// other internal service call in this project (asset-service takes
/// creatorId as given, etc.) -- this endpoint trusts whatever `viewer` the
/// caller claims. In the real stack that caller is api-gateway, which has
/// already verified the JWT and knows whether the request is actually
/// authenticated; nothing here re-checks that itself.
async fn sign_url(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SignRequest>,
) -> impl IntoResponse {
    let variant = req.viewer.variant();
    let (exp, sig) = signing::sign(
        &state.signing_secret,
        &req.artwork_id,
        variant,
        state.sign_ttl_seconds,
    );
    let url = format!(
        "/asset/{}/render?variant={}&exp={}&sig={}",
        req.artwork_id, variant, exp, sig
    );
    Json(SignResponse { url })
}

#[derive(Deserialize)]
struct RenderQuery {
    variant: String,
    exp: u64,
    sig: String,
}

#[derive(Deserialize)]
struct AssetVersion {
    #[serde(rename = "variantName")]
    variant_name: String,
    #[serde(rename = "storageUri")]
    storage_uri: String,
}

#[derive(Deserialize)]
struct ArtworkDetail {
    #[serde(rename = "assetVersions")]
    asset_versions: Vec<AssetVersion>,
}

async fn render_asset(
    State(state): State<Arc<AppState>>,
    Path(artwork_id): Path<String>,
    Query(q): Query<RenderQuery>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> Response {
    if let Err(err) = signing::verify(
        &state.signing_secret,
        &artwork_id,
        &q.variant,
        q.exp,
        &q.sig,
    ) {
        let msg = match err {
            signing::TokenError::Expired => "token expired",
            signing::TokenError::BadSignature => "invalid token",
        };
        return (StatusCode::FORBIDDEN, msg).into_response();
    }

    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if crawlers::is_known_ai_crawler(user_agent) {
        // PROJECT_DESIGN.md \u{00a7}3-5 offers "차단 또는 decoy" -- decoy/honeypot
        // responses are Phase 4 scope (Nightshade-style honeypot assets),
        // not built here; blocking outright is the real defense today.
        return (StatusCode::FORBIDDEN, "known AI crawler, blocked").into_response();
    }

    if !state.allowed_referer_hosts.is_empty() {
        if let Some(referer) = headers.get("referer").and_then(|v| v.to_str().ok()) {
            let referer_host = referer
                .parse::<http::Uri>()
                .ok()
                .and_then(|u| u.host().map(|h| h.to_string()))
                .unwrap_or_default();
            if !state
                .allowed_referer_hosts
                .iter()
                .any(|h| h == &referer_host)
            {
                return (
                    StatusCode::FORBIDDEN,
                    "referer not allowed (hotlink blocked)",
                )
                    .into_response();
            }
        }
        // No Referer header at all is allowed through -- direct navigation
        // and privacy-respecting browsers routinely strip it; treating
        // "absent" the same as "disallowed" would break normal use, not
        // just hotlinking.
    }

    let client_ip: IpAddr = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(addr.ip());
    if !state.rate_limiter.check(client_ip) {
        return (StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded").into_response();
    }

    let detail_url = format!("{}/artworks/{}", state.asset_service_url, artwork_id);
    let detail: ArtworkDetail = match state.http.get(&detail_url).send().await {
        Ok(res) if res.status().is_success() => match res.json().await {
            Ok(body) => body,
            Err(_) => {
                return (StatusCode::BAD_GATEWAY, "malformed asset-service response")
                    .into_response();
            }
        },
        Ok(res) if res.status() == reqwest::StatusCode::NOT_FOUND => {
            return (StatusCode::NOT_FOUND, "no such artwork").into_response();
        }
        _ => return (StatusCode::BAD_GATEWAY, "asset-service unreachable").into_response(),
    };

    let version = match detail
        .asset_versions
        .iter()
        .find(|v| v.variant_name == q.variant)
    {
        Some(v) => v,
        None => {
            return (
                StatusCode::NOT_FOUND,
                "variant not generated for this artwork",
            )
                .into_response();
        }
    };

    let bytes = match tokio::fs::read(&version.storage_uri).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::NOT_FOUND, "variant file missing on disk").into_response(),
    };

    let content_type = if version.storage_uri.ends_with(".png") {
        "image/png"
    } else {
        "image/jpeg"
    };
    (
        [
            ("content-type", content_type),
            // PROJECT_DESIGN.md \u{00a7}3-5: cooperative no-index signal per response,
            // same reasoning as robots.txt -- not enforcement, a courtesy.
            ("x-robots-tag", "noindex, noimageindex"),
            ("cache-control", "private, max-age=60"),
        ],
        bytes,
    )
        .into_response()
}
