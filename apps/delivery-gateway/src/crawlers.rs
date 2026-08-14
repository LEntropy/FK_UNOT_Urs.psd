//! AI-crawler denylist (PROJECT_DESIGN.md §3-5). Shared between the
//! `robots.txt` generator (cooperative, no enforcement power) and the
//! `/asset/:id/render` handler's real access control (the actual defense --
//! robots.txt is a courtesy, not a gate).
//!
//! Names/substrings taken from each crawler's own published User-Agent
//! documentation as of this writing; matching is case-insensitive substring
//! containment, not exact equality, since real UA strings carry extra
//! version/platform tokens around the identifying token.

use serde::{Deserialize, Serialize};

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

// -- Bot classification (per-artwork ALLOW/BLOCK/LOG_ONLY policy support) --
//
// Broader than AI_CRAWLER_USER_AGENTS/is_known_ai_crawler above (AI-training
// bots only, hardcoded 403 in render_asset): covers search engines,
// AI-training bots, AI-search bots, AI-assistant bots, social-preview bots,
// and SEO bots, grouped so src/bot_policy.rs can apply a configurable
// action per group (or per-bot override) instead of a single hardcoded
// block. The old table/function above are left in place rather than
// removed -- nothing else in the crate currently calls them, but removing
// call sites this merge didn't add felt like the wrong risk to take here.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BotGroup {
    SearchEngine,
    AiTraining,
    AiSearch,
    AiAssistant,
    SocialPreview,
    Seo,
    UnknownBot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BotIdentity {
    pub name: &'static str,
    pub group: BotGroup,
}

const BOTS: &[(&str, &str, BotGroup)] = &[
    ("Googlebot", "Googlebot", BotGroup::SearchEngine),
    ("Bingbot", "bingbot", BotGroup::SearchEngine),
    ("YandexBot", "YandexBot", BotGroup::SearchEngine),
    ("Baiduspider", "Baiduspider", BotGroup::SearchEngine),
    ("GPTBot", "GPTBot", BotGroup::AiTraining),
    ("Google-Extended", "Google-Extended", BotGroup::AiTraining),
    ("ClaudeBot", "ClaudeBot", BotGroup::AiTraining),
    ("CCBot", "CCBot", BotGroup::AiTraining),
    ("Bytespider", "Bytespider", BotGroup::AiTraining),
    ("Diffbot", "Diffbot", BotGroup::AiTraining),
    ("OAI-SearchBot", "OAI-SearchBot", BotGroup::AiSearch),
    ("Claude-SearchBot", "Claude-SearchBot", BotGroup::AiSearch),
    ("PerplexityBot", "PerplexityBot", BotGroup::AiSearch),
    ("ChatGPT-User", "ChatGPT-User", BotGroup::AiAssistant),
    ("Claude-User", "Claude-User", BotGroup::AiAssistant),
    ("Perplexity-User", "Perplexity-User", BotGroup::AiAssistant),
    ("Twitterbot", "Twitterbot", BotGroup::SocialPreview),
    (
        "FacebookExternalHit",
        "facebookexternalhit",
        BotGroup::SocialPreview,
    ),
    ("LinkedInBot", "LinkedInBot", BotGroup::SocialPreview),
    ("AhrefsBot", "AhrefsBot", BotGroup::Seo),
    ("SemrushBot", "SemrushBot", BotGroup::Seo),
    ("MJ12bot", "MJ12bot", BotGroup::Seo),
];

pub fn representative_user_agents() -> impl Iterator<Item = &'static str> {
    BOTS.iter().map(|(_, ua, _)| *ua)
}

pub fn classify(user_agent: &str) -> Option<BotIdentity> {
    let lower = user_agent.to_ascii_lowercase();
    if let Some((name, _, group)) = BOTS
        .iter()
        .find(|(_, token, _)| lower.contains(&token.to_ascii_lowercase()))
    {
        return Some(BotIdentity { name, group: *group });
    }
    [
        "bot",
        "crawler",
        "spider",
        "scrapy",
        "headless",
        "curl/",
        "wget/",
        "python-requests",
    ]
    .iter()
    .any(|m| lower.contains(m))
    .then_some(BotIdentity {
        name: "UNKNOWN_BOT",
        group: BotGroup::UnknownBot,
    })
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

    #[test]
    fn classifies_bots() {
        assert_eq!(
            classify("GPTBot/1.2").unwrap().group,
            BotGroup::AiTraining
        );
        assert_eq!(
            classify("my-crawler").unwrap().group,
            BotGroup::UnknownBot
        );
        assert!(classify("Mozilla Chrome").is_none());
    }
}
