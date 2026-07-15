//! Distinct-artwork enumeration detection (PHASE4_SCOPING.md's "adaptive
//! anti-scrape" -- sequential-ID-enumeration was the item recommended to
//! build first). **Adapted from the original scoping text**, not built
//! as literally described there: this project's artwork IDs
//! (`asset-service/src/routes/artworks.ts`: `ast_${randomUUID()...}`) are
//! random 16-hex-char strings, not sequential integers -- there is no
//! "ast_1, ast_2, ast_3" pattern to detect, by design (random IDs already
//! resist guessing/enumeration attacks on their own).
//!
//! The signal that *does* apply here is the same underlying behavior the
//! scoping doc was really after: a real user's session touches a handful
//! of artworks (whatever the UI actually links to); a scraper touches many
//! *distinct* artworks quickly regardless of whether the IDs are
//! sequential or random, because it's enumerating a feed/sitemap/guessed
//! list rather than browsing. Tracking distinct-artwork-count per
//! fingerprint in a sliding window captures that, without depending on an
//! ID scheme this project doesn't have.

use dashmap::DashMap;
use std::collections::HashSet;
use std::net::IpAddr;
use std::time::{Duration, Instant};

struct Window {
    seen: HashSet<String>,
    // Not per-artwork timestamps (unlike RateLimiter) -- the whole set
    // resets when the window elapses, since what matters here is "how many
    // distinct artworks in this window," not a precise sliding count.
    window_start: Instant,
}

pub struct EnumerationDetector {
    window: Duration,
    max_distinct: usize,
    state: DashMap<IpAddr, Window>,
}

impl EnumerationDetector {
    pub fn new(max_distinct: usize, window: Duration) -> Self {
        Self {
            window,
            max_distinct,
            state: DashMap::new(),
        }
    }

    /// Records a request for `artwork_id` from `ip` and returns whether
    /// the request pattern still looks like normal browsing. Once an IP
    /// trips this in a window, it stays tripped for the rest of that
    /// window (no un-flagging mid-window) -- an enumeration pass doesn't
    /// stop being one just because the next request happens to repeat an
    /// already-seen ID.
    pub fn check(&self, ip: IpAddr, artwork_id: &str) -> bool {
        let now = Instant::now();
        let mut entry = self.state.entry(ip).or_insert_with(|| Window {
            seen: HashSet::new(),
            window_start: now,
        });

        if now.duration_since(entry.window_start) >= self.window {
            entry.seen.clear();
            entry.window_start = now;
        }

        entry.seen.insert(artwork_id.to_string());
        entry.seen.len() <= self.max_distinct
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn ip() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))
    }

    #[test]
    fn allows_browsing_a_few_distinct_artworks() {
        let d = EnumerationDetector::new(5, Duration::from_secs(60));
        for id in ["a", "b", "c"] {
            assert!(d.check(ip(), id));
        }
    }

    #[test]
    fn repeatedly_requesting_the_same_artwork_never_trips_it() {
        let d = EnumerationDetector::new(2, Duration::from_secs(60));
        for _ in 0..10 {
            assert!(d.check(ip(), "same_artwork"));
        }
    }

    #[test]
    fn flags_a_burst_of_many_distinct_artworks() {
        let d = EnumerationDetector::new(3, Duration::from_secs(60));
        assert!(d.check(ip(), "a"));
        assert!(d.check(ip(), "b"));
        assert!(d.check(ip(), "c"));
        assert!(!d.check(ip(), "d")); // 4th distinct artwork this window
    }

    #[test]
    fn different_ips_are_independent() {
        let d = EnumerationDetector::new(1, Duration::from_secs(60));
        assert!(d.check(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)), "a"));
        assert!(d.check(IpAddr::V4(Ipv4Addr::new(2, 2, 2, 2)), "a"));
    }

    #[test]
    fn resets_after_the_window_elapses() {
        let d = EnumerationDetector::new(1, Duration::from_millis(50));
        assert!(d.check(ip(), "a"));
        assert!(!d.check(ip(), "b"));
        std::thread::sleep(Duration::from_millis(60));
        assert!(d.check(ip(), "c"));
    }
}
