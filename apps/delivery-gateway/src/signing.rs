//! Signed-URL tokens (PROJECT_DESIGN.md §3-5: "영구 URL 금지 → 정책 기반
//! signed URL"). HMAC-SHA256 over `{artworkId}|{variant}|{exp}`, not a JWT
//! -- there's no need for a header/claims envelope here, just a short,
//! tamper-evident, time-boxed capability token bound to one specific
//! (artwork, variant) pair.

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenError {
    Expired,
    BadSignature,
}

fn message(artwork_id: &str, variant: &str, exp: u64) -> String {
    format!("{artwork_id}|{variant}|{exp}")
}

fn hmac_hex(secret: &str, msg: &str) -> String {
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(msg.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before 1970")
        .as_secs()
}

/// Signs `(artwork_id, variant)` with a `ttl_seconds` expiry from now.
/// Returns `(exp, signature_hex)` -- both go into the URL as query params.
pub fn sign(secret: &str, artwork_id: &str, variant: &str, ttl_seconds: u64) -> (u64, String) {
    let exp = now_unix() + ttl_seconds;
    (exp, hmac_hex(secret, &message(artwork_id, variant, exp)))
}

/// Verifies a token against the current time. Checks expiry *before*
/// signature so an expired-but-validly-signed token gives a distinct error
/// from a tampered one -- useful for the caller to log/respond differently
/// (expired is normal churn; a bad signature is a tamper attempt worth
/// noting).
pub fn verify(
    secret: &str,
    artwork_id: &str,
    variant: &str,
    exp: u64,
    sig_hex: &str,
) -> Result<(), TokenError> {
    if exp < now_unix() {
        return Err(TokenError::Expired);
    }
    let expected = hmac_hex(secret, &message(artwork_id, variant, exp));
    // Constant-time compare -- hex::encode output is ASCII, safe to compare
    // as bytes; subtle isn't pulled in as a dep just for this one check.
    if constant_time_eq(expected.as_bytes(), sig_hex.as_bytes()) {
        Ok(())
    } else {
        Err(TokenError::BadSignature)
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_freshly_signed_token_verifies() {
        let (exp, sig) = sign("secret", "ast_1", "public_preview_1280", 60);
        assert!(verify("secret", "ast_1", "public_preview_1280", exp, &sig).is_ok());
    }

    #[test]
    fn an_expired_token_is_rejected_even_with_the_right_signature() {
        let (exp, sig) = sign("secret", "ast_1", "public_preview_1280", 0);
        // ttl=0 means exp == now; sleep 1s so it's unambiguously in the past
        // by the time verify() runs (avoids a same-second flake).
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert_eq!(
            verify("secret", "ast_1", "public_preview_1280", exp, &sig),
            Err(TokenError::Expired)
        );
    }

    #[test]
    fn a_tampered_signature_is_rejected() {
        let (exp, mut sig) = sign("secret", "ast_1", "public_preview_1280", 60);
        // Guaranteed to actually change the byte regardless of what it
        // started as -- replacing with a fixed "00" was a real, if rare
        // (1/256), flake: if the genuine signature's first byte already
        // was 0x00, the "tamper" was a no-op and the test passed for the
        // wrong reason.
        let replacement = if &sig[0..2] == "00" { "ff" } else { "00" };
        sig.replace_range(0..2, replacement);
        assert_eq!(
            verify("secret", "ast_1", "public_preview_1280", exp, &sig),
            Err(TokenError::BadSignature)
        );
    }

    #[test]
    fn a_token_signed_for_a_different_variant_is_rejected() {
        let (exp, sig) = sign("secret", "ast_1", "public_preview_2048", 60);
        // Same artwork, same exp, same sig, but requesting a different
        // variant than what was actually signed -- this is exactly the
        // "don't let a signed 1280 URL be replayed to fetch 2048" property.
        assert_eq!(
            verify("secret", "ast_1", "public_preview_1280", exp, &sig),
            Err(TokenError::BadSignature)
        );
    }

    #[test]
    fn a_token_signed_for_a_different_artwork_is_rejected() {
        let (exp, sig) = sign("secret", "ast_1", "public_preview_1280", 60);
        assert_eq!(
            verify("secret", "ast_2", "public_preview_1280", exp, &sig),
            Err(TokenError::BadSignature)
        );
    }
}
