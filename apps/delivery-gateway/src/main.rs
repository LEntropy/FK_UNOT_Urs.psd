use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use delivery_gateway::enumeration::EnumerationDetector;
use delivery_gateway::honeypot::HoneypotTracker;
use delivery_gateway::rate_limit::RateLimiter;
use delivery_gateway::{AppState, build_router};

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        signing_secret: std::env::var("DELIVERY_SIGNING_SECRET")
            .expect("DELIVERY_SIGNING_SECRET must be set (see .env.example)"),
        asset_service_url: std::env::var("ASSET_SERVICE_URL")
            .unwrap_or_else(|_| "http://localhost:3002".into()),
        allowed_referer_hosts: std::env::var("ALLOWED_REFERER_HOSTS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        http: reqwest::Client::new(),
        rate_limiter: RateLimiter::new(
            std::env::var("RATE_LIMIT_MAX_REQUESTS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60),
            Duration::from_secs(
                std::env::var("RATE_LIMIT_WINDOW_SECONDS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(60),
            ),
        ),
        enumeration_detector: EnumerationDetector::new(
            std::env::var("ENUMERATION_MAX_DISTINCT_ARTWORKS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30),
            Duration::from_secs(
                std::env::var("ENUMERATION_WINDOW_SECONDS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(60),
            ),
        ),
        honeypot: HoneypotTracker::new(honeypot_tokens()),
        sign_ttl_seconds: std::env::var("SIGN_TTL_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300),
    });

    let app = build_router(state);
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4500);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("delivery-gateway listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .expect("server error");
}

/// HONEYPOT_TOKENS (comma-separated) if explicitly set; otherwise one
/// random token is generated at startup -- honeypots work with zero
/// configuration, and unlike a signing secret, a honeypot token doesn't
/// need to stay stable across restarts to keep working (the only place it
/// needs to match anything is this same process's own robots.txt output,
/// generated from the same value).
fn honeypot_tokens() -> Vec<String> {
    let configured: Vec<String> = std::env::var("HONEYPOT_TOKENS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if !configured.is_empty() {
        return configured;
    }

    use sha2::{Digest, Sha256};
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before 1970")
        .as_nanos()
        .to_le_bytes();
    let mut hasher = Sha256::new();
    hasher.update(seed);
    hasher.update(std::process::id().to_le_bytes());
    vec![hex::encode(&hasher.finalize()[..8])]
}
