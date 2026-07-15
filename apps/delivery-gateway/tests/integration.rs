//! Real end-to-end tests against the actual axum router (via `tower::ServiceExt::oneshot`,
//! no mocked handler internals) with a mocked asset-service (via `wiremock`,
//! a real HTTP server on a random local port) -- proves the full request
//! path: signature verification, crawler blocking, referer checks, rate
//! limiting, and the real HTTP call out to asset-service, not just each
//! piece in isolation.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use delivery_gateway::enumeration::EnumerationDetector;
use delivery_gateway::rate_limit::RateLimiter;
use delivery_gateway::{AppState, build_router};
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SECRET: &str = "test-secret";

fn state_with_mock_asset_service(mock_server_uri: &str) -> Arc<AppState> {
    build_state(
        mock_server_uri,
        vec![],
        RateLimiter::new(1000, Duration::from_secs(60)),
        EnumerationDetector::new(1000, Duration::from_secs(60)),
    )
}

fn build_state(
    asset_service_url: &str,
    allowed_referer_hosts: Vec<String>,
    rate_limiter: RateLimiter,
    enumeration_detector: EnumerationDetector,
) -> Arc<AppState> {
    Arc::new(AppState {
        signing_secret: SECRET.to_string(),
        asset_service_url: asset_service_url.to_string(),
        allowed_referer_hosts,
        http: reqwest::Client::new(),
        rate_limiter,
        enumeration_detector,
        sign_ttl_seconds: 300,
    })
}

fn signed_render_uri(artwork_id: &str, variant: &str, ttl: u64) -> String {
    let (exp, sig) = delivery_gateway::signing::sign(SECRET, artwork_id, variant, ttl);
    format!("/asset/{artwork_id}/render?variant={variant}&exp={exp}&sig={sig}")
}

async fn send(app: axum::Router, req: Request<Body>) -> axum::http::Response<Body> {
    app.oneshot(req).await.unwrap()
}

fn request(uri: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .extension(axum::extract::ConnectInfo(SocketAddr::from((
            [127, 0, 0, 1],
            12345,
        ))))
        .body(Body::empty())
        .unwrap()
}

fn request_with_ua(uri: &str, ua: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .header("user-agent", ua)
        .extension(axum::extract::ConnectInfo(SocketAddr::from((
            [127, 0, 0, 1],
            12345,
        ))))
        .body(Body::empty())
        .unwrap()
}

#[tokio::test]
async fn health_check() {
    let state = state_with_mock_asset_service("http://unused");
    let app = build_router(state);
    let res = send(app, request("/health")).await;
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn robots_txt_denies_known_ai_crawlers_and_allows_general_indexing_off_for_asset_paths() {
    let state = state_with_mock_asset_service("http://unused");
    let app = build_router(state);
    let res = send(app, request("/robots.txt")).await;
    assert_eq!(res.status(), StatusCode::OK);

    let body = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("Disallow: /asset/"));
    assert!(text.contains("User-agent: GPTBot"));
    assert!(text.contains("User-agent: ClaudeBot"));
}

#[tokio::test]
async fn sign_then_render_returns_the_real_image_bytes_from_asset_service() {
    let mock_server = MockServer::start().await;
    let tmp_file = std::env::temp_dir().join("delivery_gateway_test_image.png");
    tokio::fs::write(&tmp_file, b"fake-png-bytes")
        .await
        .unwrap();

    Mock::given(method("GET"))
        .and(path("/artworks/ast_1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "assetVersions": [
                { "variantName": "public_preview_1280", "storageUri": tmp_file.to_str().unwrap() }
            ]
        })))
        .mount(&mock_server)
        .await;

    let state = state_with_mock_asset_service(&mock_server.uri());
    let app = build_router(state.clone());

    let sign_res = send(
        app.clone(),
        Request::builder()
            .method("POST")
            .uri("/internal/sign")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"artworkId": "ast_1", "viewer": "anonymous"}).to_string(),
            ))
            .unwrap(),
    )
    .await;
    assert_eq!(sign_res.status(), StatusCode::OK);
    let sign_body: serde_json::Value =
        serde_json::from_slice(&sign_res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let signed_url = sign_body["url"].as_str().unwrap();
    assert!(signed_url.contains("variant=public_preview_1280"));

    let render_res = send(app, request(signed_url)).await;
    assert_eq!(render_res.status(), StatusCode::OK);
    assert_eq!(
        render_res.headers().get("x-robots-tag").unwrap(),
        "noindex, noimageindex"
    );
    let body = render_res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&body[..], b"fake-png-bytes");

    tokio::fs::remove_file(&tmp_file).await.ok();
}

#[tokio::test]
async fn expired_token_is_rejected() {
    let state = state_with_mock_asset_service("http://unused");
    let app = build_router(state);
    let uri = signed_render_uri("ast_1", "public_preview_1280", 0);
    tokio::time::sleep(Duration::from_secs(1)).await;
    let res = send(app, request(&uri)).await;
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn tampered_signature_is_rejected() {
    let state = state_with_mock_asset_service("http://unused");
    let app = build_router(state);
    let mut uri = signed_render_uri("ast_1", "public_preview_1280", 60);
    uri = uri.replace("sig=", "sig=ff");
    let res = send(app, request(&uri)).await;
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn known_ai_crawler_is_blocked_even_with_a_validly_signed_token() {
    let mock_server = MockServer::start().await;
    let state = state_with_mock_asset_service(&mock_server.uri());
    let app = build_router(state);
    let uri = signed_render_uri("ast_1", "public_preview_1280", 60);
    let res = send(
        app,
        request_with_ua(
            &uri,
            "Mozilla/5.0 (compatible; GPTBot/1.0; +https://openai.com/gptbot)",
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn mismatched_referer_is_blocked_when_an_allowlist_is_configured() {
    let mock_server = MockServer::start().await;
    let state = build_state(
        &mock_server.uri(),
        vec!["dontai.example".to_string()],
        RateLimiter::new(1000, Duration::from_secs(60)),
        EnumerationDetector::new(1000, Duration::from_secs(60)),
    );
    let app = build_router(state);

    let uri = signed_render_uri("ast_1", "public_preview_1280", 60);
    let req = Request::builder()
        .uri(&uri)
        .header("referer", "https://evil-hotlinker.example/page")
        .extension(axum::extract::ConnectInfo(SocketAddr::from((
            [127, 0, 0, 1],
            12345,
        ))))
        .body(Body::empty())
        .unwrap();
    let res = send(app, req).await;
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn missing_referer_is_allowed_even_with_an_allowlist_configured() {
    let mock_server = MockServer::start().await;
    let tmp_file = std::env::temp_dir().join("delivery_gateway_test_image2.png");
    tokio::fs::write(&tmp_file, b"fake-png-bytes-2")
        .await
        .unwrap();
    Mock::given(method("GET"))
        .and(path("/artworks/ast_2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "assetVersions": [{ "variantName": "public_preview_1280", "storageUri": tmp_file.to_str().unwrap() }]
        })))
        .mount(&mock_server)
        .await;

    let state = build_state(
        &mock_server.uri(),
        vec!["dontai.example".to_string()],
        RateLimiter::new(1000, Duration::from_secs(60)),
        EnumerationDetector::new(1000, Duration::from_secs(60)),
    );
    let app = build_router(state);

    let uri = signed_render_uri("ast_2", "public_preview_1280", 60);
    let res = send(app, request(&uri)).await;
    assert_eq!(res.status(), StatusCode::OK);

    tokio::fs::remove_file(&tmp_file).await.ok();
}

#[tokio::test]
async fn rate_limit_blocks_after_the_configured_threshold() {
    let mock_server = MockServer::start().await;
    let state = build_state(
        &mock_server.uri(),
        vec![],
        RateLimiter::new(2, Duration::from_secs(60)),
        EnumerationDetector::new(1000, Duration::from_secs(60)),
    );
    let app = build_router(state);

    let uri = signed_render_uri("ast_1", "public_preview_1280", 60);
    // Both requests use the same signed token and the same connection IP --
    // the limiter keys on IP, not token, so this is a fair test of "too
    // many requests from one client" regardless of what they're requesting.
    let r1 = send(app.clone(), request(&uri)).await;
    let r2 = send(app.clone(), request(&uri)).await;
    let r3 = send(app, request(&uri)).await;

    // r1/r2 may 502 (no asset-service mock for this path) or 200 -- what
    // matters here is only that the 3rd is rate-limited, not their status.
    assert_ne!(r1.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_ne!(r2.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(r3.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn browsing_a_few_distinct_artworks_is_never_flagged_as_enumeration() {
    let mock_server = MockServer::start().await;
    let state = build_state(
        &mock_server.uri(),
        vec![],
        RateLimiter::new(1000, Duration::from_secs(60)),
        EnumerationDetector::new(2, Duration::from_secs(60)),
    );
    let app = build_router(state);

    for id in ["ast_1", "ast_2"] {
        let uri = signed_render_uri(id, "public_preview_1280", 60);
        let res = send(app.clone(), request(&uri)).await;
        assert_ne!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}

#[tokio::test]
async fn requesting_many_distinct_artworks_quickly_is_flagged_as_enumeration() {
    let mock_server = MockServer::start().await;
    let state = build_state(
        &mock_server.uri(),
        vec![],
        RateLimiter::new(1000, Duration::from_secs(60)),
        EnumerationDetector::new(2, Duration::from_secs(60)),
    );
    let app = build_router(state);

    let ids = ["ast_1", "ast_2", "ast_3"];
    let mut last_status = StatusCode::OK;
    for id in ids {
        let uri = signed_render_uri(id, "public_preview_1280", 60);
        last_status = send(app.clone(), request(&uri)).await.status();
    }
    // The 3rd *distinct* artwork this IP has touched within the window
    // trips the max_distinct=2 threshold, regardless of what asset-service
    // would have said about that artwork.
    assert_eq!(last_status, StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn repeatedly_re_requesting_the_same_artwork_never_trips_enumeration() {
    let mock_server = MockServer::start().await;
    let tmp_file = std::env::temp_dir().join("delivery_gateway_test_image3.png");
    tokio::fs::write(&tmp_file, b"fake-png-bytes-3")
        .await
        .unwrap();
    Mock::given(method("GET"))
        .and(path("/artworks/ast_repeat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "assetVersions": [{ "variantName": "public_preview_1280", "storageUri": tmp_file.to_str().unwrap() }]
        })))
        .mount(&mock_server)
        .await;

    let state = build_state(
        &mock_server.uri(),
        vec![],
        RateLimiter::new(1000, Duration::from_secs(60)),
        EnumerationDetector::new(1, Duration::from_secs(60)),
    );
    let app = build_router(state);

    let uri = signed_render_uri("ast_repeat", "public_preview_1280", 60);
    for _ in 0..5 {
        let res = send(app.clone(), request(&uri)).await;
        assert_eq!(res.status(), StatusCode::OK);
    }

    tokio::fs::remove_file(&tmp_file).await.ok();
}

#[tokio::test]
async fn requesting_a_variant_the_artwork_never_generated_is_a_404_not_a_500() {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/artworks/ast_no_variant"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "assetVersions": [] })),
        )
        .mount(&mock_server)
        .await;

    let state = state_with_mock_asset_service(&mock_server.uri());
    let app = build_router(state);
    let uri = signed_render_uri("ast_no_variant", "public_preview_1280", 60);
    let res = send(app, request(&uri)).await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}
