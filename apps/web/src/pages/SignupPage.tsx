import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuthStore } from "../store/auth";

export function SignupPage() {
  const navigate = useNavigate();
  const login = useAuthStore((s) => s.login);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [handle, setHandle] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await api.post<{ accessToken: string; refreshToken: string; user: any }>("/auth/signup", {
        email,
        password,
        handle,
      });
      login(res.accessToken, res.refreshToken, res.user);
      navigate("/");
    } catch (err) {
      setError(err instanceof ApiError ? describeError(err) : "회원가입에 실패했습니다.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto mt-16 max-w-sm">
      <h1 className="mb-6 text-2xl font-semibold">회원가입</h1>
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
          type="text"
          required
          placeholder="핸들 (영문/숫자/_)"
          value={handle}
          onChange={(e) => setHandle(e.target.value)}
          className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
        />
        <input
          type="password"
          required
          placeholder="비밀번호 (8자 이상)"
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
          {submitting ? "가입 중..." : "가입하기"}
        </button>
      </form>
      <p className="mt-4 text-sm text-neutral-400">
        이미 계정이 있으신가요?{" "}
        <Link to="/login" className="underline">
          로그인
        </Link>
      </p>
    </div>
  );
}

function describeError(err: ApiError): string {
  if (err.status === 409) return "이미 사용 중인 이메일 또는 핸들입니다.";
  if (err.status === 400) return "입력값을 확인해주세요.";
  return "회원가입에 실패했습니다.";
}
