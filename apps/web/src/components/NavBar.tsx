import { Link, NavLink, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useAuthStore } from "../store/auth";
import { useCoinBalance } from "../hooks/useCoinBalance";
import { getMe } from "../api/users";
import { Avatar } from "./Avatar";
import {
  HomeIcon,
  GridIcon,
  UploadIcon,
  FlaskIcon,
  ShieldIcon,
  UserIcon,
  SettingsIcon,
  CoinIcon,
  LogoutIcon,
} from "./icons";

// X/Pixiv-style persistent nav (2026-08-14 redesign): a left icon+label
// sidebar on desktop, a fixed icon-only bottom bar on mobile -- replacing
// the old single-row text-link bar that had no real navigation hierarchy
// (no profile, no settings, no visual weight on the primary action). One
// component renders both layouts via responsive classes rather than two
// separate components, so the nav item list only has to be maintained once.
export function NavBar() {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const navigate = useNavigate();
  const { balance } = useCoinBalance();

  const { data: me } = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    enabled: !!user,
    staleTime: 60 * 1000,
  });

  if (!user) {
    // Logged-out surfaces (login/signup/terms) get a minimal top bar only --
    // the full nav assumes an authenticated identity to point avatar/coin/
    // settings at.
    return (
      <nav className="flex items-center justify-between border-b border-neutral-800 px-6 py-3">
        <Link to="/" className="text-lg font-semibold tracking-tight text-neutral-50">
          DONTAI
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link to="/login" className="text-neutral-300 hover:text-neutral-100">
            로그인
          </Link>
          <Link to="/signup" className="btn-primary">
            회원가입
          </Link>
          <Link to="/terms" className="text-neutral-500 hover:underline">
            이용약관
          </Link>
        </div>
      </nav>
    );
  }

  const displayName = me?.displayName || user.handle;

  const navItem = (to: string, label: string, Icon: (p: { className?: string }) => JSX.Element, end = false) => (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `flex items-center gap-4 rounded-full px-4 py-2.5 text-[15px] font-medium transition-colors ${
          isActive ? "bg-neutral-900 text-brand-400" : "text-neutral-300 hover:bg-neutral-900 hover:text-neutral-50"
        }`
      }
    >
      <Icon className="h-6 w-6 shrink-0" />
      <span className="hidden xl:inline">{label}</span>
    </NavLink>
  );

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-20 flex-col justify-between border-r border-neutral-800 bg-neutral-950 px-2 py-4 md:flex xl:w-64 xl:px-4">
        <div className="flex flex-col gap-1">
          <Link to="/" className="mb-3 flex items-center gap-2 rounded-full px-4 py-2 text-xl font-bold text-neutral-50">
            <span className="text-brand-500">●</span>
            <span className="hidden xl:inline">DONTAI</span>
          </Link>
          {navItem("/", "홈", HomeIcon, true)}
          {navItem("/my-artworks", "내 작품", GridIcon)}
          {navItem("/upload", "업로드", UploadIcon)}
          {navItem("/test-lab", "테스트 랩", FlaskIcon)}
          {(user.role === "MODERATOR" || user.role === "ADMIN") && navItem("/moderation", "모더레이션", ShieldIcon)}
          {navItem(`/creators/${user.id}`, "프로필", UserIcon)}
          {navItem("/settings", "설정", SettingsIcon)}
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 rounded-full bg-amber-950/40 px-4 py-2 text-sm text-amber-300">
            <CoinIcon className="h-5 w-5 shrink-0" />
            <span className="hidden xl:inline">{balance !== null ? `코인 ${balance}` : "..."}</span>
          </div>
          <Link
            to={`/creators/${user.id}`}
            className="flex items-center gap-3 rounded-full px-2 py-2 hover:bg-neutral-900"
          >
            <Avatar seed={user.id} label={displayName} avatarUri={me?.avatarUri} size="sm" />
            <div className="hidden min-w-0 flex-1 xl:block">
              <p className="truncate text-sm font-semibold text-neutral-100">{displayName}</p>
              <p className="truncate text-xs text-neutral-500">@{user.handle}</p>
            </div>
          </Link>
          <button
            onClick={() => {
              logout();
              navigate("/login");
            }}
            className="flex items-center gap-4 rounded-full px-4 py-2.5 text-sm text-neutral-400 hover:bg-neutral-900 hover:text-neutral-100"
          >
            <LogoutIcon className="h-5 w-5 shrink-0" />
            <span className="hidden xl:inline">로그아웃</span>
          </button>
        </div>
      </aside>

      {/* Mobile top bar */}
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-neutral-800 bg-neutral-950/95 px-4 py-3 backdrop-blur md:hidden">
        <Link to="/" className="text-lg font-bold text-neutral-50">
          DONTAI
        </Link>
        <div className="flex items-center gap-3">
          <span className="rounded-full bg-amber-950/40 px-3 py-1 text-xs text-amber-300">🪙 {balance ?? "..."}</span>
          <Link to={`/creators/${user.id}`}>
            <Avatar seed={user.id} label={displayName} avatarUri={me?.avatarUri} size="sm" />
          </Link>
        </div>
      </header>

      {/* Mobile bottom tab bar */}
      <nav className="fixed inset-x-0 bottom-0 z-30 flex items-center justify-around border-t border-neutral-800 bg-neutral-950/95 py-1.5 backdrop-blur md:hidden">
        <MobileTab to="/" Icon={HomeIcon} end />
        <MobileTab to="/my-artworks" Icon={GridIcon} />
        <MobileTab to="/upload" Icon={UploadIcon} />
        <MobileTab to="/test-lab" Icon={FlaskIcon} />
        <MobileTab to="/settings" Icon={SettingsIcon} />
      </nav>
    </>
  );
}

function MobileTab({ to, Icon, end }: { to: string; Icon: (p: { className?: string }) => JSX.Element; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) => `rounded-full p-2.5 ${isActive ? "text-brand-400" : "text-neutral-400"}`}
    >
      <Icon className="h-6 w-6" />
    </NavLink>
  );
}
