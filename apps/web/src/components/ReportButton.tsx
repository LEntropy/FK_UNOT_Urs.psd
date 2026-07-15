import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import * as community from "../api/community";

export function ReportButton({ artworkId }: { artworkId: string }) {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState("");

  const report = useMutation({
    mutationFn: () => community.reportArtwork(artworkId, reason),
  });

  // Success is checked before the `!open` early-return -- closing the form
  // on success (setOpen(false)) would otherwise make this branch
  // unreachable, showing the plain "신고" button again instead of
  // confirming the report actually went through.
  if (report.isSuccess) {
    return <p className="text-sm text-neutral-500">신고가 접수되었습니다.</p>;
  }

  if (!open) {
    return (
      <button onClick={() => setOpen(true)} className="text-sm text-neutral-500 hover:text-red-400">
        신고
      </button>
    );
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (reason.trim()) report.mutate();
      }}
      className="flex flex-col gap-2 rounded border border-neutral-800 p-3"
    >
      <textarea
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        placeholder="신고 사유를 입력하세요"
        maxLength={1000}
        rows={2}
        className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm"
      />
      <div className="flex gap-2">
        <button
          type="submit"
          disabled={report.isPending || !reason.trim()}
          className="rounded bg-red-900 px-3 py-1 text-sm text-red-200 disabled:opacity-50"
        >
          신고 제출
        </button>
        <button type="button" onClick={() => setOpen(false)} className="text-sm text-neutral-500">
          취소
        </button>
      </div>
    </form>
  );
}
