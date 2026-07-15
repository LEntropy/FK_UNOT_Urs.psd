import { Link, useNavigate } from "react-router-dom";
import { useAuthStore } from "../store/auth";

export function NavBar() {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const navigate = useNavigate();

  return (
    <nav className="flex items-center justify-between border-b border-neutral-800 px-6 py-3">
      <Link to="/" className="text-lg font-semibold tracking-tight">
        DONTAI
      </Link>
      <div className="flex items-center gap-4 text-sm">
        {user ? (
          <>
            <Link to="/" className="hover:underline">
              피드
            </Link>
            <Link to="/upload" className="hover:underline">
              업로드
            </Link>
            <Link to="/my-artworks" className="hover:underline">
              내 작품
            </Link>
            {(user.role === "MODERATOR" || user.role === "ADMIN") && (
              <Link to="/moderation" className="hover:underline">
                모더레이션
              </Link>
            )}
            <span className="text-neutral-400">@{user.handle}</span>
            <button
              onClick={() => {
                logout();
                navigate("/login");
              }}
              className="rounded bg-neutral-800 px-3 py-1 hover:bg-neutral-700"
            >
              로그아웃
            </button>
          </>
        ) : (
          <>
            <Link to="/login" className="hover:underline">
              로그인
            </Link>
            <Link to="/signup" className="hover:underline">
              회원가입
            </Link>
          </>
        )}
      </div>
    </nav>
  );
}
