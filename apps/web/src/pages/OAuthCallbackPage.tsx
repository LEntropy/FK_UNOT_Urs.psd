import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuthStore } from "../store/auth";

/** Lands here from api-gateway's /auth/:provider/callback redirect, which puts
 * tokens in the URL hash (not the query string) so they never hit server logs. */
export function OAuthCallbackPage() {
  const navigate = useNavigate();
  const login = useAuthStore((s) => s.login);
  const setAccessToken = useAuthStore((s) => s.setAccessToken);
  const [error, setError] = useState<string | null>(null);
  const ran = useRef(false);

  useEffect(() => {
    if (ran.current) return;
    ran.current = true;

    const params = new URLSearchParams(window.location.hash.slice(1));
    const accessToken = params.get("accessToken");
    const refreshToken = params.get("refreshToken");

    if (!accessToken || !refreshToken) {
      setError("로그인 응답에서 토큰을 찾을 수 없습니다.");
      return;
    }

    // /me needs an accessToken in the store before it can be called.
    setAccessToken(accessToken);

    api
      .get<{ id: string; email: string; handle: string; walletAddress: string; role: string }>("/me")
      .then((user) => {
        login(accessToken, refreshToken, user);
        navigate("/", { replace: true });
      })
      .catch((err) => {
        useAuthStore.getState().logout(); // clear the partial accessToken set above
        setError(err instanceof ApiError ? "로그인 정보를 불러오지 못했습니다." : "알 수 없는 오류가 발생했습니다.");
      });
  }, [login, setAccessToken, navigate]);

  return (
    <div className="mx-auto mt-16 max-w-sm text-center">
      {error ? (
        <>
          <p className="text-sm text-red-400">{error}</p>
          <button onClick={() => navigate("/login")} className="mt-4 underline">
            로그인으로 돌아가기
          </button>
        </>
      ) : (
        <p className="text-sm text-neutral-400">로그인 처리 중...</p>
      )}
    </div>
  );
}
