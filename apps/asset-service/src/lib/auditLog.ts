import { createHash } from "node:crypto";
import type { Db } from "../db/client.js";
import { complianceAuditLogs } from "../db/schema.js";

/** Appends one row to compliance_audit_logs (see schema.ts's own doc for
 * why this table exists and how it differs from this project's other,
 * non-append-only logs). Call this for consent/rights-relevant state
 * changes only -- not every request, just the ones that would matter if a
 * Do-Not-Train claim or takedown ever got legally contested (upload-time
 * consent, bot access policy changes, original-preview availability
 * changes). `payload` is hashed, not stored raw -- the append-only
 * guarantee (drizzle/0013's SQLite triggers) is only meaningful for a
 * fixed-shape hash, not a payload whose shape can drift across app
 * versions. */
export function logComplianceAudit(
  db: Db,
  entry: {
    userWallet: string | null;
    actionType: string;
    targetArtworkId: string | null;
    ipAddress: string | null;
    userAgent: string | null;
    payload: unknown;
  },
): void {
  const payloadHash = createHash("sha256").update(JSON.stringify(entry.payload)).digest("hex");
  db.insert(complianceAuditLogs)
    .values({
      userWallet: entry.userWallet,
      actionType: entry.actionType,
      targetArtworkId: entry.targetArtworkId,
      ipAddress: entry.ipAddress,
      userAgent: entry.userAgent,
      payloadHash,
      createdAt: new Date(),
    })
    .run();
}
