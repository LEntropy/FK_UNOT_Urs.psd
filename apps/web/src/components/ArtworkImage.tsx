import { useQuery } from "@tanstack/react-query";
import * as delivery from "../api/delivery";

/**
 * Fetches a signed, short-TTL render URL from api-gateway (which is the
 * only trusted caller of delivery-gateway's /internal/sign -- see that
 * service's README) and points an <img> straight at delivery-gateway with
 * it. No permanent image URL exists anywhere in this app, matching
 * PROJECT_DESIGN.md §3-5's "영구 URL 금지" -- a stale cached URL just
 * expires (delivery-gateway's SIGN_TTL_SECONDS) rather than needing to be
 * revoked.
 */
export function ArtworkImage({
  artworkId,
  hasVariants,
  variant = "logged_in",
  className,
  /** Feed/gallery grids resolve every card's URL in one batched request
   * (api/delivery.ts's getRenderUrlsBatch) and pass the result straight
   * through here -- skips this component's own per-artwork fetch entirely.
   * Undefined value with batchMode=false (the default) keeps the original
   * single-fetch behavior for standalone uses (artwork detail page's main
   * image). */
  preloadedSrc,
  /** 2026-08-14 fix: batchMode is a separate flag from preloadedSrc's own
   * value on purpose -- while the batch request is still in flight,
   * ArtworkGrid passes preloadedSrc=undefined (nothing resolved yet), which
   * used to be indistinguishable from "no batch in use at all" and made
   * this component fire its OWN redundant per-artwork request during that
   * window -- every card in a grid briefly double-requested its image (once
   * from its own query, once from the batch), needlessly multiplying
   * delivery-gateway's request count right as rate-limit/enumeration
   * thresholds (src/rate_limit.rs, src/enumeration.rs there) are the most
   * likely to matter. With batchMode=true this component never fires its
   * own query at all -- it just shows a skeleton until preloadedSrc
   * resolves to a real value (or null on failure). */
  batchMode = false,
}: {
  artworkId: string;
  hasVariants: boolean;
  variant?: "logged_in" | "thumbnail" | "original";
  className?: string;
  preloadedSrc?: string | null;
  batchMode?: boolean;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["renderUrl", artworkId, variant],
    queryFn: () => delivery.getRenderUrl(artworkId, variant),
    enabled: hasVariants && !batchMode,
    // Signed URLs expire (delivery-gateway default: 5 minutes) -- refetch
    // a fresh one periodically rather than letting a long-open tab's <img>
    // silently start 403ing.
    staleTime: 4 * 60 * 1000,
    refetchInterval: 4 * 60 * 1000,
  });

  if (!hasVariants) {
    return (
      <div className={`flex items-center justify-center bg-neutral-900 text-xs text-neutral-500 ${className}`}>
        보호 처리 중
      </div>
    );
  }

  const src = batchMode ? preloadedSrc : data?.url;

  if (!batchMode && isLoading) {
    return <div className={`skeleton ${className}`} />;
  }
  if ((!batchMode && (error || !data)) || src === null) {
    return <div className={`flex items-center justify-center bg-neutral-900 text-xs text-red-400 ${className}`}>이미지를 불러오지 못했습니다</div>;
  }
  if (!src) {
    return <div className={`skeleton ${className}`} />;
  }

  return <img src={src} alt="" loading="lazy" decoding="async" className={className} />;
}
