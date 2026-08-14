pub mod bot_policy;
pub mod crawlers;
pub mod enumeration;
pub mod honeypot;
pub mod honeypot_db;
pub mod rate_limit;
pub mod signing;

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bot_policy::{ArtworkBotPolicy, BotAction, BotPolicyStore};
use enumeration::EnumerationDetector;
use honeypot::HoneypotTracker;
use rate_limit::RateLimiter;
use serde::{Deserialize, Serialize};

pub struct AppState {
    pub signing_secret: String,
    pub asset_service_url: String,
    pub allowed_referer_hosts: Vec<String>,
    pub http: reqwest::Client,
    pub rate_limiter: RateLimiter,
    pub enumeration_detector: EnumerationDetector,
    pub honeypot: HoneypotTracker,
    pub sign_ttl_seconds: u64,
    pub bot_policies: BotPolicyStore,
}

/// Loads every artwork's persisted bot policy from asset-service (the
/// system of record, schema.ts's botPolicies table) into this process's
/// own cache once at startup -- without this, a restart would silently
/// reset every creator's ALLOW/BLOCK/LOG_ONLY choice back to the built-in
/// default until someone happened to PUT it again (put_bot_policy's own
/// write-through keeps it correct going forward, but does nothing for
/// policies set before this restart). Best-effort: asset-service being
/// unreachable at startup (e.g. this process wins a race to start first)
/// logs a warning and leaves the cache empty rather than blocking startup
/// or crashing -- individual policies still work correctly once GET/PUT
/// bot-policy calls touch them, same as this cache always behaved before
/// this hydration step existed.
pub async fn hydrate_bot_policies(state: &Arc<AppState>) {
    let url = format!("{}/artworks/bot-policies", state.asset_service_url);
    let rows: Vec<serde_json::Value> = match state.http.get(&url).send().await {
        Ok(res) if res.status().is_success() => match res.json().await {
            Ok(rows) => rows,
            Err(err) => {
                eprintln!("bot-policy hydration: malformed asset-service response ({err}), starting with an empty cache");
                return;
            }
        },
        Ok(res) => {
            eprintln!(
                "bot-policy hydration: asset-service returned {}, starting with an empty cache",
                res.status()
            );
            return;
        }
        Err(err) => {
            eprintln!("bot-policy hydration: asset-service unreachable ({err}), starting with an empty cache");
            return;
        }
    };

    let mut loaded = 0usize;
    for row in rows {
        let Some(artwork_id) = row.get("artworkId").and_then(|v| v.as_str()) else { continue };
        match serde_json::from_value::<ArtworkBotPolicy>(row.clone()) {
            Ok(policy) => {
                state.bot_policies.set(artwork_id.to_string(), policy);
                loaded += 1;
            }
            Err(err) => eprintln!("bot-policy hydration: skipping {artwork_id} ({err})"),
        }
    }
    println!("bot-policy hydration: loaded {loaded} artwork polic{}", if loaded == 1 { "y" } else { "ies" });
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/health",
            get(|| async { Json(serde_json::json!({"status": "ok"})) }),
        )
        .route("/robots.txt", get(robots_txt))
        .route("/internal/sign", post(sign_url))
        .route("/internal/honeypot-hits", get(honeypot_hits))
        .route(
            "/internal/bot-policies/{id}",
            get(get_bot_policy).put(put_bot_policy),
        )
        .route("/internal/bot-access-logs", get(bot_access_logs))
        .route("/asset/{id}/render", get(render_asset))
        .route("/decoy/{token}", get(decoy))
        .with_state(state)
}

async fn robots_txt(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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

    if !state.honeypot.tokens().is_empty() {
        body.push_str(
            "\n# PHASE4_SCOPING.md \u{00a7}2 honeypot URLs: no real page in this app ever\n\
             # links to these -- listing them here is the *only* place they're\n\
             # mentioned. A real hit is either a crawler ignoring Disallow, or one\n\
             # scraping robots.txt for \"interesting\" paths -- not a human.\n",
        );
        for token in state.honeypot.tokens() {
            body.push_str(&format!("Disallow: /decoy/{token}\n"));
        }
    }

    ([("content-type", "text/plain; charset=utf-8")], body)
}

/// Serves a real, valid, 200-OK decoy image and logs the hit -- never a
/// 403/404, since tipping the scraper off would stop it generating more
/// signal. See src/honeypot.rs's module doc for the full reasoning.
async fn decoy(
    State(state): State<Arc<AppState>>,
    Path(token): Path<String>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    if state.honeypot.is_honeypot_token(&token) {
        let user_agent = headers
            .get("user-agent")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let client_ip: IpAddr = headers
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(',').next())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(addr.ip());
        state.honeypot.record_hit(&token, client_ip, user_agent);
    }
    ([("content-type", "image/png")], honeypot::DECOY_PNG_1X1)
}

/// Ops-only introspection, not meant to be public -- no auth of its own
/// (matching every other "internal" endpoint's trust boundary in this
/// project), a real deployment would put this behind a private network or
/// its own auth, not expose it the way /asset/:id/render is meant to be.
async fn honeypot_hits(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.honeypot.recent_hits(100))
}

/// Per-artwork ALLOW/BLOCK/LOG_ONLY bot policy, read by render_asset's
/// classify()+action() check below. Same "internal, no auth of its own"
/// trust boundary as the other /internal/* endpoints in this file.
async fn get_bot_policy(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    Json(state.bot_policies.get(&id))
}

/// Write-through, not write-back: persists to asset-service (the system of
/// record, see schema.ts's botPolicies doc) BEFORE updating this process's
/// own cache, and only updates the cache if that succeeds -- keeps the two
/// from silently diverging if asset-service is unreachable, rather than
/// having a policy change "work" locally and then vanish on the next
/// restart. Same asset_service_url + state.http this AppState already uses
/// for render_asset's own artwork-detail lookup.
async fn put_bot_policy(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(policy): Json<ArtworkBotPolicy>,
) -> impl IntoResponse {
    let url = format!("{}/artworks/{}/bot-policy", state.asset_service_url, id);
    match state.http.put(&url).json(&policy).send().await {
        Ok(res) if res.status().is_success() => {
            state.bot_policies.set(id, policy.clone());
            (StatusCode::OK, Json(policy)).into_response()
        }
        Ok(res) => {
            let status = res.status().as_u16();
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("asset-service rejected bot-policy write (status {status})")})),
            )
                .into_response()
        }
        Err(_) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": "asset-service unreachable"})),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LogQuery {
    artwork_id: Option<String>,
    limit: Option<usize>,
}

/// Best-effort, fire-and-forget durable log push (2026-08-14) -- render_
/// asset's own response to the actual requester must never wait on or fail
/// because of this. A dropped log entry here just means one row missing
/// from the creator-facing history; the in-memory tail (BotPolicyStore's
/// own ring buffer, still recorded synchronously before this is called)
/// stays correct either way. Same "spawn a task, ignore the result" pattern
/// this project already uses for other non-critical persistence (see
/// remote_gpu.py's best-effort remote tmp-file cleanup for the Python-side
/// equivalent reasoning).
fn spawn_bot_access_log_push(state: Arc<AppState>, entry: bot_policy::BotAccessLog) {
    tokio::spawn(async move {
        let url = format!("{}/artworks/bot-access-logs", state.asset_service_url);
        let body = serde_json::json!({
            "artworkId": entry.artwork_id,
            "botName": entry.bot_name,
            "botGroup": entry.group,
            "action": entry.action,
            "clientIp": entry.client_ip,
            "userAgent": entry.user_agent,
            "responseStatus": entry.response_status,
            "timestampUnixSeconds": entry.timestamp,
        });
        let _ = state.http.post(&url).json(&body).send().await;
    });
}

async fn bot_access_logs(
    State(state): State<Arc<AppState>>,
    Query(q): Query<LogQuery>,
) -> impl IntoResponse {
    Json(
        state
            .bot_policies
            .recent(q.artwork_id.as_deref(), q.limit.unwrap_or(100)),
    )
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
    // Coin-system feature (2026-08-10): the creator-opt-in, coin-unlocked
    // near-original derivative (asset-service's POST /:id/original-preview,
    // ml-engine/src/original_preview.py). Access control lives entirely on
    // asset-service's side (does the "original_preview" asset_versions row
    // exist, has this viewer unlocked it) -- api-gateway only requests this
    // variant after that check already passed, same trust boundary as
    // every other Viewer case here.
    OriginalPreview,
}

impl Viewer {
    /// PROJECT_DESIGN.md \u{00a7}3-5: "비로그인 → 1280px, 로그인 유저 → 2048px".
    /// `Thumbnail` isn't in the original design text -- used by the feed/
    /// gallery grid, where fast loading matters more than resolution.
    /// Points at feed_thumbnail_original (rust-core variants.rs's
    /// generate_feed_thumbnail: a small, original-sourced image, not a
    /// downscale of the protected one -- see that function's own doc for
    /// why serving the original is safe at this resolution) rather than
    /// grid_thumbnail_512, which is both bigger (slower to load in a grid
    /// of many artworks) and, for STRONG_PROTECTION artworks specifically,
    /// visibly distorted (a known issue, tracked separately).
    /// render_asset's own fallback-to-largest-available logic below still
    /// covers artworks from before this variant existed.
    fn variant(self) -> &'static str {
        match self {
            Viewer::Anonymous => "public_preview_1280",
            Viewer::LoggedIn => "public_preview_2048",
            Viewer::Thumbnail => "feed_thumbnail_original",
            Viewer::OriginalPreview => "original_preview",
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
    width: i64,
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
    let client_ip: IpAddr = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(addr.ip());

    // Policy-driven bot handling (src/bot_policy.rs): classify the request
    // via crawlers::classify(), then resolve this artwork's configured
    // action for that bot/group instead of a single hardcoded block. Not a
    // regression from the old `is_known_ai_crawler` 403 -- the default
    // policy (ArtworkBotPolicy::default) still blocks the AiTraining group,
    // matching prior behavior, while extending coverage to search/social/
    // SEO/AI-search/AI-assistant bots (logged, not blocked, by default) and
    // letting a creator override per artwork.
    if let Some(bot) = crawlers::classify(user_agent) {
        let action = state.bot_policies.action(&artwork_id, &bot);
        if action == BotAction::Block {
            // PROJECT_DESIGN.md \u{00a7}3-5 offers "차단 또는 decoy" -- decoy/honeypot
            // responses are Phase 4 scope (Nightshade-style honeypot assets),
            // not built here; blocking outright is the real defense today.
            let entry = state
                .bot_policies
                .record(&artwork_id, &bot, action, client_ip, user_agent, 403);
            spawn_bot_access_log_push(state.clone(), entry);
            return (StatusCode::FORBIDDEN, "bot blocked by artwork policy").into_response();
        }
        let entry = state
            .bot_policies
            .record(&artwork_id, &bot, action, client_ip, user_agent, 200);
        spawn_bot_access_log_push(state.clone(), entry);
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

    // PHASE4_SCOPING.md §2's own recommendation, previously unimplemented:
    // "a honeypot hit should immediately and permanently flag that
    // fingerprint, not just nudge a score." A hit on /decoy/:token is
    // unambiguous by construction (see honeypot.rs's module doc) -- no
    // real user can ever reach it, so there's no false-positive risk in
    // blocking that IP outright, unlike the rate limiter/enumeration
    // detector below, which use soft, resettable thresholds because they
    // reason about ambiguous signals a real heavy user could trip.
    if state.honeypot.is_flagged(client_ip) {
        return (
            StatusCode::FORBIDDEN,
            "blocked -- this IP already tripped a honeypot",
        )
            .into_response();
    }
    if !state.rate_limiter.check(client_ip) {
        return (StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded").into_response();
    }
    // PHASE4_SCOPING.md's adaptive-anti-scrape recommendation, adapted to
    // this project's random (non-sequential) artwork IDs -- see
    // src/enumeration.rs's module doc for why. A normal session touches a
    // handful of artworks; systematically fetching many distinct ones
    // quickly looks like enumeration regardless of ID scheme.
    if !state.enumeration_detector.check(client_ip, &artwork_id) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "too many distinct artworks requested -- looks like enumeration",
        )
            .into_response();
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
        // rust-core deliberately never upscales (variants.rs's own
        // skips_variants_that_would_upscale) -- a modest-resolution real
        // upload can genuinely have no public_preview_2048 or even
        // grid_thumbnail_512 variant, only whatever's smaller than the
        // source. The signed token authorizes "this artwork, this
        // requested tier, this expiry," not a guarantee that exact tier
        // exists -- falling back to the largest variant that *does* exist
        // serves something real instead of leaving the whole page with a
        // broken image over a request nobody could have satisfied exactly.
        .or_else(|| detail.asset_versions.iter().max_by_key(|v| v.width))
    {
        Some(v) => v,
        None => {
            return (
                StatusCode::NOT_FOUND,
                "no variants generated for this artwork yet",
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
