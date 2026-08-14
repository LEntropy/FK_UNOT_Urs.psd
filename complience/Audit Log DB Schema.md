CREATE TABLE compliance_audit_logs (
    log_id BIGSERIAL PRIMARY KEY,
    user_wallet VARCHAR(42) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    target_asset_id VARCHAR(50),
    ip_address VARCHAR(45) NOT NULL,
    user_agent TEXT NOT NULL,
    payload_hash VARCHAR(64) NOT NULL,
    kms_signature TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Append-Only 통제 (수정/삭제 금지 트리거)
CREATE RULE prevent_audit_update AS ON UPDATE TO compliance_audit_logs DO INSTEAD NOTHING;
CREATE RULE prevent_audit_delete AS ON DELETE TO compliance_audit_logs DO INSTEAD NOTHING;