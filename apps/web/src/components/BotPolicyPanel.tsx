import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BOT_GROUPS,
  BOT_GROUP_LABEL,
  getBotPolicy,
  setBotPolicy,
  getBotAccessLogs,
  type ArtworkBotPolicy,
  type BotAction,
} from "../api/botPolicy";

const ACTION_LABEL: Record<BotAction, string> = { ALLOW: "허용", BLOCK: "차단", LOG_ONLY: "기록만" };

/** Creator-only: per-artwork bot ALLOW/BLOCK/LOG_ONLY policy + recent access
 * history (motection-changes handoff feature, DB-backed as of 2026-08-14 --
 * see delivery-gateway/src/bot_policy.rs and asset-service schema.ts's
 * botPolicies table doc for the persistence side). Group-level control
 * only in this UI (not individual bot-name overrides) -- that covers the
 * actual decision a creator cares about ("should AI-training bots reach my
 * art"); per-bot fine-tuning is still possible via the API, just not
 * surfaced here. */
export function BotPolicyPanel({ artworkId }: { artworkId: string }) {
  const { data: policy, refetch } = useQuery({
    queryKey: ["botPolicy", artworkId],
    queryFn: () => getBotPolicy(artworkId),
  });
  const { data: logs } = useQuery({
    queryKey: ["botAccessLogs", artworkId],
    queryFn: () => getBotAccessLogs(artworkId, 20),
  });

  const [draft, setDraft] = useState<ArtworkBotPolicy | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState(false);

  useEffect(() => {
    if (policy) setDraft(policy);
  }, [policy]);

  async function onSave() {
    if (!draft) return;
    setSaving(true);
    setError(null);
    try {
      await setBotPolicy(artworkId, draft);
      await refetch();
    } catch {
      setError("봇 정책을 저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }

  if (!draft) return null;

  return (
    <div className="mb-6 rounded border border-neutral-800 px-4 py-3">
      <p className="mb-2 text-sm font-medium">봇 접근 관리</p>
      <div className="mb-3 flex flex-wrap gap-x-6 gap-y-2 text-sm">
        {BOT_GROUPS.map((group) => (
          <label key={group} className="flex items-center gap-2">
            <span className="text-neutral-400">{BOT_GROUP_LABEL[group]}</span>
            <select
              value={draft.groupPolicies[group] ?? draft.defaultAction}
              onChange={(e) =>
                setDraft({
                  ...draft,
                  groupPolicies: { ...draft.groupPolicies, [group]: e.target.value as BotAction },
                })
              }
              className="rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs"
            >
              {(Object.keys(ACTION_LABEL) as BotAction[]).map((a) => (
                <option key={a} value={a}>
                  {ACTION_LABEL[a]}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={saving}
          onClick={() => void onSave()}
          className="rounded bg-neutral-100 px-3 py-2 text-sm font-medium text-neutral-900 hover:bg-neutral-300 disabled:opacity-50"
        >
          정책 저장
        </button>
        <button type="button" onClick={() => setShowLogs((v) => !v)} className="text-xs text-neutral-400 underline">
          {showLogs ? "접근 기록 숨기기" : "최근 접근 기록 보기"}
        </button>
      </div>
      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
      {showLogs && (
        <div className="mt-3 max-h-64 overflow-y-auto text-xs">
          {!logs || logs.length === 0 ? (
            <p className="text-neutral-500">아직 기록된 봇 접근이 없습니다.</p>
          ) : (
            <table className="w-full text-left">
              <thead className="text-neutral-500">
                <tr>
                  <th className="pr-3 font-normal">시각</th>
                  <th className="pr-3 font-normal">봇</th>
                  <th className="pr-3 font-normal">처리</th>
                  <th className="font-normal">IP 대역</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => (
                  <tr key={log.id} className="text-neutral-300">
                    <td className="pr-3 py-0.5">{new Date(log.timestamp).toLocaleString()}</td>
                    <td className="pr-3 py-0.5">{log.botName}</td>
                    <td className="pr-3 py-0.5">{ACTION_LABEL[log.action]}</td>
                    <td className="py-0.5 text-neutral-500">{log.clientIp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
