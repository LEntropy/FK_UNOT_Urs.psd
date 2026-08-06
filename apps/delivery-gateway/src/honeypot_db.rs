//! SQLite persistence for HoneypotTracker (PHASE4_SCOPING.md §2's own
//! "worth building future work" note: the hit log was in-memory-only,
//! meaning a restart both lost the evidence trail AND silently unblocked
//! every IP that had already tripped a honeypot -- the second one is a
//! real regression, not just lost history, since `is_flagged` is what
//! `render_asset` uses to keep a caught scraper blocked.
//!
//! Deliberately NOT on `is_flagged`'s hot path (every real request calls
//! that): this module only backs `record_hit` (write-through) and startup
//! hydration of the in-memory `DashMap`/`DashSet` HoneypotTracker already
//! uses for the actual per-request check -- see honeypot.rs's
//! `with_persistence` constructor. A honeypot hit itself is a rare,
//! attacker-triggered event, not something on any real user's request
//! path, so a blocking SQLite write there is fine.

use rusqlite::Connection;
use std::net::IpAddr;
use std::sync::Mutex;

use crate::honeypot::HoneypotHit;

pub struct HoneypotDb {
    conn: Mutex<Connection>,
}

impl HoneypotDb {
    pub fn open(path: &str) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS honeypot_hits (
                id INTEGER PRIMARY KEY,
                token TEXT NOT NULL,
                ip TEXT NOT NULL,
                user_agent TEXT NOT NULL,
                unix_time INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS flagged_ips (
                ip TEXT PRIMARY KEY
            );",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn insert_hit(&self, hit: &HoneypotHit) {
        let conn = self.conn.lock().expect("honeypot db mutex poisoned");
        // Best-effort like everything else in this pipeline's enrichment
        // steps -- a DB write failure (disk full, etc) shouldn't stop
        // record_hit from still flagging the IP in memory for this
        // process's own lifetime.
        let _ = conn.execute(
            "INSERT INTO honeypot_hits (token, ip, user_agent, unix_time) VALUES (?1, ?2, ?3, ?4)",
            (&hit.token, &hit.ip, &hit.user_agent, hit.unix_time),
        );
    }

    pub fn flag_ip(&self, ip: IpAddr) {
        let conn = self.conn.lock().expect("honeypot db mutex poisoned");
        let _ = conn.execute(
            "INSERT OR IGNORE INTO flagged_ips (ip) VALUES (?1)",
            [ip.to_string()],
        );
    }

    /// Loads every persisted hit, oldest first -- used once at startup to
    /// hydrate the in-memory hit log so `GET /internal/honeypot-hits`
    /// still has history after a restart.
    pub fn load_all_hits(&self) -> Vec<HoneypotHit> {
        let conn = self.conn.lock().expect("honeypot db mutex poisoned");
        let mut stmt = match conn.prepare("SELECT token, ip, user_agent, unix_time FROM honeypot_hits ORDER BY id ASC") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map([], |row| {
            Ok(HoneypotHit {
                token: row.get(0)?,
                ip: row.get(1)?,
                user_agent: row.get(2)?,
                unix_time: row.get(3)?,
            })
        });
        match rows {
            Ok(rows) => rows.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Loads every IP ever flagged -- used once at startup so a restart
    /// doesn't silently un-block an already-caught scraper.
    pub fn load_flagged_ips(&self) -> Vec<IpAddr> {
        let conn = self.conn.lock().expect("honeypot db mutex poisoned");
        let mut stmt = match conn.prepare("SELECT ip FROM flagged_ips") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map([], |row| row.get::<_, String>(0));
        match rows {
            Ok(rows) => rows
                .filter_map(Result::ok)
                .filter_map(|s| s.parse::<IpAddr>().ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn round_trips_hits_and_flagged_ips_through_a_real_sqlite_file() {
        let dir = std::env::temp_dir().join(format!("honeypot_db_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("honeypot.db");
        let path_str = path.to_str().unwrap();
        let _ = std::fs::remove_file(&path);

        let db = HoneypotDb::open(path_str).unwrap();
        db.insert_hit(&HoneypotHit {
            token: "abc123".to_string(),
            ip: "1.1.1.1".to_string(),
            user_agent: "bot-1".to_string(),
            unix_time: 1_700_000_000,
        });
        db.flag_ip(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)));

        // Reopen (simulates a restart) -- must still see the same data.
        let db2 = HoneypotDb::open(path_str).unwrap();
        let hits = db2.load_all_hits();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].ip, "1.1.1.1");

        let flagged = db2.load_flagged_ips();
        assert_eq!(flagged, vec![IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn flagging_the_same_ip_twice_does_not_duplicate() {
        let dir = std::env::temp_dir().join(format!("honeypot_db_test2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("honeypot.db");
        let path_str = path.to_str().unwrap();
        let _ = std::fs::remove_file(&path);

        let db = HoneypotDb::open(path_str).unwrap();
        let ip = IpAddr::V4(Ipv4Addr::new(2, 2, 2, 2));
        db.flag_ip(ip);
        db.flag_ip(ip);

        assert_eq!(db.load_flagged_ips().len(), 1);

        std::fs::remove_file(&path).ok();
    }
}
