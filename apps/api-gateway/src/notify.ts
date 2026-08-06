import nodemailer from "nodemailer";

/**
 * Sends the "증거 확보됨" email a creator gets when detection-svc's
 * background scan/report/model-leak pipeline finds real evidence against
 * their artwork. Best-effort, same posture as evidenceSigning.ts's own
 * failure handling and everything else in detection-svc's evidence
 * pipeline (screenshot/PDF/C2PA): SMTP not configured (no credentials in
 * this dev/demo environment) or a delivery failure both degrade to "no
 * email sent" rather than throwing -- POST /internal/notify-evidence-ready
 * always returns 200 either way, so a notification failure can never turn
 * an already-successful case back into an error for detection-svc.
 */

let cachedTransport: nodemailer.Transporter | null | undefined;

function getTransport(): nodemailer.Transporter | null {
  if (cachedTransport !== undefined) return cachedTransport;

  const host = process.env.SMTP_HOST;
  const user = process.env.SMTP_USER;
  const pass = process.env.SMTP_PASSWORD;
  if (!host || !user || !pass) {
    cachedTransport = null;
    return null;
  }

  cachedTransport = nodemailer.createTransport({
    host,
    port: process.env.SMTP_PORT ? Number(process.env.SMTP_PORT) : 587,
    secure: process.env.SMTP_SECURE === "1",
    auth: { user, pass },
  });
  return cachedTransport;
}

export interface EvidenceReadyNotification {
  toEmail: string;
  artworkTitle: string;
  caseId: string;
  evidenceType: "copy" | "model_leak";
}

export async function sendEvidenceReadyEmail(notification: EvidenceReadyNotification): Promise<boolean> {
  const transport = getTransport();
  if (!transport) return false;

  const subject =
    notification.evidenceType === "model_leak"
      ? `[DONTAI] "${notification.artworkTitle}" 학습 유출 의심 증거가 확보됐습니다`
      : `[DONTAI] "${notification.artworkTitle}" 무단 재배포 증거가 확보됐습니다`;

  try {
    await transport.sendMail({
      from: process.env.SMTP_FROM || "no-reply@dontai.local",
      to: notification.toEmail,
      subject,
      text: `케이스 ${notification.caseId}에서 실제 증거가 확보됐습니다. DONTAI 대시보드의 테스트 랩 > 증빙·추적 탭에서 케이스 상세를 확인해주세요.`,
    });
    return true;
  } catch {
    return false;
  }
}

/** Test-only escape hatch -- resets the cached transport so a test can
 * flip SMTP_* env vars between cases without a stale transport lingering. */
export function _resetTransportCacheForTests(): void {
  cachedTransport = undefined;
}
