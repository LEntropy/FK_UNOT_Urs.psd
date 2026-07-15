//! AI-crawler denylist (PROJECT_DESIGN.md §3-5). Shared between the
//! `robots.txt` generator (cooperative, no enforcement power) and the
//! `/asset/:id/render` handler's real access control (the actual defense --
//! robots.txt is a courtesy, not a gate).
//!
//! Names/substrings taken from each crawler's own published User-Agent
//! documentation as of this writing; matching is case-insensitive substring
//! containment, not exact equality, since real UA strings carry extra
//! version/platform tokens around the identifying token.

pub const AI_CRAWLER_USER_AGENTS: &[&str] = &[
    "GPTBot",
    "ChatGPT-User",
    "OAI-SearchBot",
    "Google-Extended",
    "ClaudeBot",
    "anthropic-ai",
    "CCBot",
    "Bytespider",
    "PerplexityBot",
    "Diffbot",
];

pub fn is_known_ai_crawler(user_agent: &str) -> bool {
    let ua_lower = user_agent.to_lowercase();
    AI_CRAWLER_USER_AGENTS
        .iter()
        .any(|known| ua_lower.contains(&known.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_crawlers_case_insensitively() {
        assert!(is_known_ai_crawler(
            "Mozilla/5.0 AppleWebKit (compatible; GPTBot/1.2; +https://openai.com/gptbot)"
        ));
        assert!(is_known_ai_crawler("claudebot/1.0"));
    }

    #[test]
    fn does_not_flag_a_normal_browser() {
        assert!(!is_known_ai_crawler(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        ));
    }
}
