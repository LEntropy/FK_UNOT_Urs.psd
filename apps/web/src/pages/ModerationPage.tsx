import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Navigate } from "react-router-dom";
import * as community from "../api/community";
import { useAuthStore } from "../store/auth";

export function ModerationPage() {
  const role = useAuthStore((s) => s.user?.role);
  const queryClient = useQueryClient();

  const { data: reports, isLoading } = useQuery({
    queryKey: ["moderationQueue"],
    queryFn: () => community.listModerationQueue("PENDING"),
    enabled: role === "MODERATOR" || role === "ADMIN",
  });

  const resolve = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "RESOLVED" | "DISMISSED" }) =>
      community.resolveReport(id, status),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["moderationQueue"] }),
  });

  // Mirrors api-gateway's own role gate -- this is a UX convenience (skip
  // rendering a page that would just 403), not the real access control.
  // The real enforcement is server-side in api-gateway's requireModerator.
  if (role !== "MODERATOR" && role !== "ADMIN") {
    return <Navigate to="/" replace />;
  }

  return (
    <div className="mx-auto mt-8 max-w-3xl">
      <h1 className="mb-6 text-2xl font-semibold">모더레이션 큐</h1>
      {isLoading && <p className="text-neutral-400">불러오는 중...</p>}
      {reports?.length === 0 && <p className="text-neutral-400">대기 중인 신고가 없습니다.</p>}
      <ul className="flex flex-col gap-3">
        {reports?.map((report) => (
          <li key={report.id} className="rounded border border-neutral-800 px-4 py-3">
            <p className="text-sm text-neutral-500">
              작품 {report.artworkId} · 신고자 {report.reporterId}
            </p>
            <p className="mt-1">{report.reason}</p>
            <div className="mt-3 flex gap-2">
              <button
                onClick={() => resolve.mutate({ id: report.id, status: "RESOLVED" })}
                disabled={resolve.isPending}
                className="rounded bg-red-900 px-3 py-1 text-sm text-red-200 disabled:opacity-50"
              >
                조치 완료
              </button>
              <button
                onClick={() => resolve.mutate({ id: report.id, status: "DISMISSED" })}
                disabled={resolve.isPending}
                className="rounded border border-neutral-700 px-3 py-1 text-sm disabled:opacity-50"
              >
                기각
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
