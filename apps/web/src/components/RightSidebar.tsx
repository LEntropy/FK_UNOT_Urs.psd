import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { SearchIcon, ActivityIcon } from "./icons";

/**
 * X/Pixiv-style right rail (2026-08-16 redesign) -- global search (was
 * previously duplicated inside FeedPage's own search box; consolidated
 * here so it's reachable from every page, not just the feed) plus a real
 * platform-stats widget and the legal footer links.
 *
 * Deliberately real numbers only, never placeholders -- this project
 * doesn't put fabricated stats in front of users (same principle as
 * STRONG_PROTECTION's UI copy honesty fix and the WORM disclosure).
 */
export function RightSidebar() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");

  const { data: stats } = useQuery({
    queryKey: ["platformStats"],
    queryFn: () => api.get<{ publishedArtworks: number; strongProtectionArtworks: number }>("/artworks/stats"),
    staleTime: 60 * 1000,
  });

  return (
    <div className="flex flex-col gap-4">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          const trimmed = query.trim();
          if (trimmed) navigate(`/?tag=${encodeURIComponent(trimmed)}`);
        }}
        className="flex items-center gap-2 rounded-full border border-neutral-800 bg-neutral-900 px-4 py-2.5 transition-colors focus-within:border-brand-600"
      >
        <SearchIcon className="h-4 w-4 shrink-0 text-neutral-500" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="태그로 작품 검색"
          className="w-full bg-transparent text-sm text-neutral-100 placeholder-neutral-500 outline-none"
        />
      </form>

      <div className="card space-y-3 p-4">
        <h3 className="flex items-center gap-2 text-sm font-bold text-neutral-200">
          <ActivityIcon className="h-4 w-4 text-emerald-400" />
          <span>플랫폼 현황</span>
        </h3>
        <div className="space-y-2 text-xs">
          <div className="flex justify-between text-neutral-400">
            <span>공개된 작품</span>
            <span className="font-bold text-neutral-100">{stats ? stats.publishedArtworks.toLocaleString() : "—"}</span>
          </div>
          <div className="flex justify-between text-neutral-400">
            <span>강력 보호 적용 작품</span>
            <span className="font-bold text-brand-400">
              {stats ? stats.strongProtectionArtworks.toLocaleString() : "—"}
            </span>
          </div>
        </div>
      </div>

      <footer className="space-y-2 px-2 text-[11px] text-neutral-500">
        <div className="flex flex-wrap gap-x-2 gap-y-1">
          <Link to="/terms" className="hover:underline">
            이용약관
          </Link>
          <span>·</span>
          <Link to="/terms" className="hover:underline">
            Do-Not-Train 정책
          </Link>
        </div>
        <p>© 2026 DONTAI</p>
      </footer>
    </div>
  );
}
