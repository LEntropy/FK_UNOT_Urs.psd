import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuthStore } from "../store/auth";
import { OAuthButtons } from "../components/OAuthButtons";

export function LoginPage() {
  const navigate = useNavigate();
  const login = useAuthStore((s) => s.login);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await api.post<{ accessToken: string; refreshToken: string; user: any }>("/auth/login", {
        email,
        password,
      });
      login(res.accessToken, res.refreshToken, res.user);
      navigate("/");
    } catch (err) {
      setError(err instanceof ApiError && err.status === 401 ? "이메일 또는 비밀번호가 올바르지 않습니다." : "로그인에 실패했습니다.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto mt-16 max-w-sm">
      <h1 className="mb-6 text-2xl font-semibold">로그인</h1>
      <form onSubmit={onSubmit} className="flex flex-col gap-3">
        <input
          type="email"
          required
          placeholder="이메일"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
        />
        <input
          type="password"
          required
          placeholder="비밀번호"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
        />
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded bg-neutral-100 px-3 py-2 font-medium text-neutral-900 disabled:opacity-50"
        >
          {submitting ? "로그인 중..." : "로그인"}
        </button>
      </form>
      <OAuthButtons />
      <p className="mt-4 text-sm text-neutral-400">
        계정이 없으신가요?{" "}
        <Link to="/signup" className="underline">
          회원가입
        </Link>
      </p>
    </div>
  );
}
