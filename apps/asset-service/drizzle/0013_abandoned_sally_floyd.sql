CREATE TABLE `compliance_audit_logs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`user_wallet` text,
	`action_type` text NOT NULL,
	`target_artwork_id` text,
	`ip_address` text,
	`user_agent` text,
	`payload_hash` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
-- Append-only enforcement (complience/Audit Log DB Schema.md's Postgres
-- draft used `CREATE RULE ... DO INSTEAD NOTHING`; SQLite has no RULE
-- statement, so this uses BEFORE triggers that abort the statement
-- instead -- same effect: UPDATE/DELETE against this table always fails).
CREATE TRIGGER `compliance_audit_logs_no_update`
BEFORE UPDATE ON `compliance_audit_logs`
BEGIN
  SELECT RAISE(ABORT, 'compliance_audit_logs is append-only');
END;
--> statement-breakpoint
CREATE TRIGGER `compliance_audit_logs_no_delete`
BEFORE DELETE ON `compliance_audit_logs`
BEGIN
  SELECT RAISE(ABORT, 'compliance_audit_logs is append-only');
END;
