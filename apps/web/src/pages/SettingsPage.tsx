import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getMe, updateMe } from "../api/users";
import { useAuthStore } from "../store/auth";
import { useCoinBalance } from "../hooks/useCoinBalance";
import { Avatar } from "../components/Avatar";

const BIO_MAX = 160;

/** Settings page (2026-08-14, new) -- the gap the user pointed at directly:
 * no way to edit a display name/bio, no account overview, nothing that
 * makes this feel like a real service instead of a fixed test account.
 * Handle/email/wallet stay read-only (see api-gateway routes/me.ts's own
 * doc for why only displayName/bio are editable here). */
export function SettingsPage() {
  const navigate = useNavigate();
  const logout = useAuthStore((s) => s.logout);
  const queryClient = useQueryClient();
  const { balance } = useCoinBalance();

  const { data: me, isLoading } = useQuery({ queryKey: ["me"], queryFn: getMe });

  const [displayName, setDisplayName] = useState("");
  const [bio, setBio] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (me) {
      setDisplayName(me.displayName ?? "");
      setBio(me.bio ?? "");
    }
  }, [me]);

  const save = useMutation({
    mutationFn: () => updateMe({ displayName: displayName.trim() || undefined, bio }),
    onSuccess: (updated) => {
      queryClient.setQueryData(["me"], updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    },
  });

  if (isLoading || !me) return <p className="mt-12 text-center text-sm text-neutral-400">불러오는 중...</p>;

  return (
    <div className="mx-auto max-w-xl pb-16">
      <h1 className="mb-6 text-xl font-bold text-neutral-50">설정</h1>

      <section className="card mb-6 p-6">
        <p className="mb-4 text-sm font-semibold text-neutral-200">프로필</p>
        <div className="mb-5 flex items-center gap-4">
          <Avatar seed={me.id} label={displayName || me.handle} avatarUri={me.avatarUri} size="lg" />
          <p className="text-xs text-neutral-500">
            아바타는 계정을 기준으로 자동 생성돼요. <br />
            직접 업로드하는 기능은 아직 준비 중입니다.
          </p>
        </div>

        <label className="mb-4 block">
          <span className="mb-1.5 block text-sm font-medium text-neutral-300">닉네임</span>
          <input
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            maxLength={50}
            placeholder={me.handle}
            className="w-full rounded-lg border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100 outline-none focus:border-brand-500"
          />
        </label>

        <label className="mb-2 block">
          <span className="mb-1.5 block text-sm font-medium text-neutral-300">자기소개</span>
          <textarea
            value={bio}
            onChange={(e) => setBio(e.target.value.slice(0, BIO_MAX))}
            rows={3}
            placeholder="나를 한 줄로 소개해보세요"
            className="w-full resize-none rounded-lg border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100 outline-none focus:border-brand-500"
          />
        </label>
        <p className="mb-4 text-right text-xs text-neutral-600">
          {bio.length} / {BIO_MAX}
        </p>

        <div className="flex items-center gap-3">
          <button onClick={() => save.mutate()} disabled={save.isPending} className="btn-primary">
            {save.isPending ? "저장 중..." : "저장"}
          </button>
          {saved && <span className="text-sm text-emerald-400">저장됐습니다</span>}
          {save.isError && <span className="text-sm text-red-400">저장에 실패했습니다</span>}
        </div>
      </section>

      <section className="card mb-6 p-6">
        <p className="mb-4 text-sm font-semibold text-neutral-200">계정 정보</p>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2.5 text-sm">
          <dt className="text-neutral-500">아이디</dt>
          <dd className="text-neutral-200">@{me.handle}</dd>
          <dt className="text-neutral-500">이메일</dt>
          <dd className="truncate text-neutral-200">{me.email}</dd>
          <dt className="text-neutral-500">지갑 주소</dt>
          <dd className="truncate font-mono text-xs text-neutral-400">{me.walletAddress}</dd>
          <dt className="text-neutral-500">코인 잔액</dt>
          <dd className="text-neutral-200">🪙 {balance ?? "..."}</dd>
          <dt className="text-neutral-500">역할</dt>
          <dd className="text-neutral-200">{me.role}</dd>
        </dl>
      </section>

      <button
        onClick={() => {
          logout();
          navigate("/login");
        }}
        className="btn-secondary w-full !text-red-300 hover:!border-red-800 hover:!bg-red-950/30"
      >
        로그아웃
      </button>
    </div>
  );
}
