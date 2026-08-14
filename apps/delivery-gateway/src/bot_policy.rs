use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::{Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use crate::crawlers::{BotGroup, BotIdentity};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BotAction { Allow, Block, LogOnly }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtworkBotPolicy {
    #[serde(default = "default_action")] pub default_action: BotAction,
    #[serde(default)] pub group_policies: HashMap<BotGroup, BotAction>,
    #[serde(default)] pub bot_overrides: HashMap<String, BotAction>,
}
fn default_action() -> BotAction { BotAction::LogOnly }
impl Default for ArtworkBotPolicy { fn default() -> Self { let mut g=HashMap::new(); g.insert(BotGroup::SearchEngine,BotAction::Allow); g.insert(BotGroup::SocialPreview,BotAction::Allow); g.insert(BotGroup::AiTraining,BotAction::Block); for x in [BotGroup::AiSearch,BotGroup::AiAssistant,BotGroup::Seo,BotGroup::UnknownBot] { g.insert(x,BotAction::LogOnly); } Self{default_action:BotAction::LogOnly,group_policies:g,bot_overrides:HashMap::new()} } }

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BotAccessLog { pub timestamp:u64, pub artwork_id:String, pub bot_name:String, pub group:BotGroup, pub action:BotAction, pub client_ip:String, pub user_agent:String, pub response_status:u16 }

pub struct BotPolicyStore { policies:RwLock<HashMap<String,ArtworkBotPolicy>>, logs:Mutex<VecDeque<BotAccessLog>>, max_logs:usize }
impl BotPolicyStore {
    pub fn new(max_logs:usize)->Self{Self{policies:RwLock::new(HashMap::new()),logs:Mutex::new(VecDeque::new()),max_logs}}
    pub fn get(&self,id:&str)->ArtworkBotPolicy{self.policies.read().unwrap().get(id).cloned().unwrap_or_default()}
    pub fn set(&self,id:String,p:ArtworkBotPolicy){self.policies.write().unwrap().insert(id,p);}
    pub fn action(&self,id:&str,b:&BotIdentity)->BotAction{let p=self.get(id);p.bot_overrides.get(b.name).copied().or_else(||p.group_policies.get(&b.group).copied()).unwrap_or(p.default_action)}
    // Returns the entry it just recorded (2026-08-14) so callers can push
    // the same already-masked-IP record to asset-service's durable log
    // (lib.rs's render_asset) without re-deriving/re-masking anything.
    pub fn record(&self,id:&str,b:&BotIdentity,a:BotAction,ip:IpAddr,ua:&str,status:u16)->BotAccessLog{let entry=BotAccessLog{timestamp:SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),artwork_id:id.into(),bot_name:b.name.into(),group:b.group,action:a,client_ip:mask_ip(ip),user_agent:ua.chars().take(512).collect(),response_status:status};let mut l=self.logs.lock().unwrap();l.push_front(entry.clone());while l.len()>self.max_logs{l.pop_back();}entry}
    pub fn recent(&self,id:Option<&str>,limit:usize)->Vec<BotAccessLog>{self.logs.lock().unwrap().iter().filter(|x|id.is_none_or(|v|x.artwork_id==v)).take(limit.min(500)).cloned().collect()}
}
fn mask_ip(ip:IpAddr)->String{match ip{IpAddr::V4(v)=>{let o=v.octets();format!("{}.{}.{}.0/24",o[0],o[1],o[2])},IpAddr::V6(_)=>"ipv6-masked".into()}}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_block_training_bots_but_only_log_unknown_bots() {
        let store = BotPolicyStore::new(10);
        let training = BotIdentity { name: "GPTBot", group: BotGroup::AiTraining };
        let unknown = BotIdentity { name: "UNKNOWN_BOT", group: BotGroup::UnknownBot };
        assert_eq!(store.action("art-1", &training), BotAction::Block);
        assert_eq!(store.action("art-1", &unknown), BotAction::LogOnly);
    }

    #[test]
    fn artwork_override_wins_and_logs_are_filtered_per_artwork() {
        let store = BotPolicyStore::new(10);
        let bot = BotIdentity { name: "GPTBot", group: BotGroup::AiTraining };
        let mut policy = ArtworkBotPolicy::default();
        policy.bot_overrides.insert("GPTBot".into(), BotAction::Allow);
        store.set("art-1".into(), policy);
        assert_eq!(store.action("art-1", &bot), BotAction::Allow);
        assert_eq!(store.action("art-2", &bot), BotAction::Block);
        store.record("art-1", &bot, BotAction::Allow, "203.0.113.42".parse().unwrap(), "GPTBot", 200);
        assert_eq!(store.recent(Some("art-1"), 10).len(), 1);
        assert!(store.recent(Some("art-2"), 10).is_empty());
    }
}
