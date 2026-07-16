//! Honeypot assets/URLs (PHASE4_SCOPING.md §2). A real user can never reach
//! `/decoy/:token` by clicking anything -- no page in this stack ever
//! links to it. The only place it's mentioned at all is `robots.txt`'s
//! `Disallow` list (`src/lib.rs`'s `robots_txt`), seeded once at startup
//! from `HONEYPOT_TOKENS`. That makes a hit here a strong signal *by
//! construction*: either a crawler that ignores `Disallow` outright, or
//! one that scrapes `robots.txt` looking for "interesting" disallowed
//! paths (a real, documented scraper behavior) -- either way, not a human
//! following a link, with zero false-positive risk from real traffic.
//!
//! Serves a real 200 with decoy image bytes rather than a `403` -- the
//! crawler-blocking path (`src/crawlers.rs`) already handles "detected,
//! turn away"; this path exists specifically to *not* tip the scraper off
//! that it's been caught, so it keeps behaving normally and every future
//! hit keeps generating signal instead of the scraper adapting.

use dashmap::DashMap;
use std::net::IpAddr;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, serde::Serialize)]
pub struct HoneypotHit {
    pub token: String,
    pub ip: String,
    pub user_agent: String,
    pub unix_time: u64,
}

pub struct HoneypotTracker {
    tokens: Vec<String>,
    hits: DashMap<u64, HoneypotHit>,
    next_id: std::sync::atomic::AtomicU64,
}

impl HoneypotTracker {
    pub fn new(tokens: Vec<String>) -> Self {
        Self {
            tokens,
            hits: DashMap::new(),
            next_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn is_honeypot_token(&self, token: &str) -> bool {
        self.tokens.iter().any(|t| t == token)
    }

    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    pub fn record_hit(&self, token: &str, ip: IpAddr, user_agent: &str) {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let unix_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before 1970")
            .as_secs();
        self.hits.insert(
            id,
            HoneypotHit {
                token: token.to_string(),
                ip: ip.to_string(),
                user_agent: user_agent.to_string(),
                unix_time,
            },
        );
    }

    pub fn recent_hits(&self, limit: usize) -> Vec<HoneypotHit> {
        let mut hits: Vec<HoneypotHit> = self
            .hits
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        hits.sort_by_key(|b| std::cmp::Reverse(b.unix_time));
        hits.truncate(limit);
        hits
    }
}

/// A tiny, real, valid 1x1 PNG -- not a placeholder string that would 500
/// trying to serve as image/png. Real bytes a browser or image-scraping
/// tool would actually decode successfully, matching "looks like it
/// worked" for the scraper.
pub const DECOY_PNG_1X1: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
    0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
    0x44, 0xAE, 0x42, 0x60, 0x82,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn recognizes_only_configured_tokens() {
        let t = HoneypotTracker::new(vec!["abc123".to_string()]);
        assert!(t.is_honeypot_token("abc123"));
        assert!(!t.is_honeypot_token("not-a-real-token"));
    }

    #[test]
    fn records_and_returns_hits_newest_first() {
        let t = HoneypotTracker::new(vec!["abc123".to_string()]);
        t.record_hit("abc123", IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)), "bot-1");
        std::thread::sleep(std::time::Duration::from_millis(1100)); // unix_time has 1s resolution
        t.record_hit("abc123", IpAddr::V4(Ipv4Addr::new(2, 2, 2, 2)), "bot-2");

        let hits = t.recent_hits(10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].ip, "2.2.2.2"); // newest first
        assert_eq!(hits[1].ip, "1.1.1.1");
    }

    #[test]
    fn recent_hits_respects_the_limit() {
        let t = HoneypotTracker::new(vec!["abc123".to_string()]);
        for i in 0..5 {
            t.record_hit("abc123", IpAddr::V4(Ipv4Addr::new(1, 1, 1, i)), "bot");
        }
        assert_eq!(t.recent_hits(3).len(), 3);
    }

    #[test]
    fn decoy_png_is_a_real_decodable_1x1_png() {
        let img = image::load_from_memory(DECOY_PNG_1X1)
            .expect("DECOY_PNG_1X1 should be a real, valid PNG");
        assert_eq!((img.width(), img.height()), (1, 1));
    }
}
