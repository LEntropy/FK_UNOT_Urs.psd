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
}: {
  artworkId: string;
  hasVariants: boolean;
  variant?: "logged_in" | "thumbnail";
  className?: string;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["renderUrl", artworkId, variant],
    queryFn: () => delivery.getRenderUrl(artworkId, variant),
    enabled: hasVariants,
    // Signed URLs expire (delivery-gateway default: 5 minutes) -- refetch
    // a fresh one periodically rather than letting a long-open tab's <img>
    // silently start 403ing.
    staleTime: 4 * 60 * 1000,
    refetchInterval: 4 * 60 * 1000,
  });

  if (!hasVariants) {
    return (
      <div className={`flex items-center justify-center bg-neutral-900 text-sm text-neutral-500 ${className}`}>
        보호 처리가 아직 완료되지 않았습니다
      </div>
    );
  }
  if (isLoading) {
    return <div className={`flex items-center justify-center bg-neutral-900 text-sm text-neutral-500 ${className}`}>불러오는 중...</div>;
  }
  if (error || !data) {
    return <div className={`flex items-center justify-center bg-neutral-900 text-sm text-red-400 ${className}`}>이미지를 불러오지 못했습니다</div>;
  }

  return <img src={data.url} alt="" className={className} />;
}
