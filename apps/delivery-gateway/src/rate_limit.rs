//! Per-IP sliding-window rate limiter (PROJECT_DESIGN.md §3-5's "rate
//! limit"). In-memory, single-process -- fine for this PoC's one instance;
//! a real multi-instance deployment would need this in Redis instead (same
//! "not built yet" scope note as everywhere else in this project that
//! currently uses in-memory state where a shared service would eventually
//! be needed).

use dashmap::DashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

pub struct RateLimiter {
    window: Duration,
    max_requests: usize,
    hits: DashMap<IpAddr, Vec<Instant>>,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            window,
            max_requests,
            hits: DashMap::new(),
        }
    }

    /// Records a request from `ip` and returns whether it's allowed.
    /// Prunes timestamps older than the window on every call rather than
    /// running a separate background sweep -- simplest correct approach at
    /// this scale (a real high-traffic deployment would want a cheaper
    /// fixed-window or token-bucket counter instead of a Vec per IP).
    pub fn check(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let mut entry = self.hits.entry(ip).or_default();
        entry.retain(|&t| now.duration_since(t) < self.window);
        if entry.len() >= self.max_requests {
            return false;
        }
        entry.push(now);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn ip() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))
    }

    #[test]
    fn allows_requests_under_the_limit() {
        let limiter = RateLimiter::new(3, Duration::from_secs(60));
        assert!(limiter.check(ip()));
        assert!(limiter.check(ip()));
        assert!(limiter.check(ip()));
    }

    #[test]
    fn blocks_requests_over_the_limit_within_the_window() {
        let limiter = RateLimiter::new(2, Duration::from_secs(60));
        assert!(limiter.check(ip()));
        assert!(limiter.check(ip()));
        assert!(!limiter.check(ip()));
    }

    #[test]
    fn different_ips_have_independent_limits() {
        let limiter = RateLimiter::new(1, Duration::from_secs(60));
        assert!(limiter.check(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
        assert!(limiter.check(IpAddr::V4(Ipv4Addr::new(2, 2, 2, 2))));
    }

    #[test]
    fn old_hits_outside_the_window_are_pruned_and_stop_counting() {
        let limiter = RateLimiter::new(1, Duration::from_millis(50));
        assert!(limiter.check(ip()));
        assert!(!limiter.check(ip()));
        std::thread::sleep(Duration::from_millis(60));
        assert!(limiter.check(ip()));
    }
}
