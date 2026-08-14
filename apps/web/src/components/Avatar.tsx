// Generated deterministic avatar (2026-08-14 redesign) -- no avatar-upload
// pipeline exists yet (users.avatarUri is reserved but unused, see
// api-gateway schema.ts), so every user gets a colorful initial-letter
// avatar derived from their own id, the same pattern Slack/Discord/GitHub
// fall back to. Deterministic (same id -> same color+letter every render,
// every session) rather than random, so a user recognizes "their" color
// across the app instead of it flickering between reloads.
const PALETTE = [
  "#f43f5e", // rose
  "#f97316", // orange
  "#eab308", // yellow
  "#22c55e", // green
  "#14b8a6", // teal
  "#3b82f6", // blue
  "#8b5cf6", // violet
  "#ec4899", // pink
];

function colorFor(seed: string): string {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = (hash << 5) - hash + seed.charCodeAt(i);
    hash |= 0;
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

const SIZE_CLASSES = {
  xs: "h-6 w-6 text-[10px]",
  sm: "h-8 w-8 text-xs",
  md: "h-10 w-10 text-sm",
  lg: "h-16 w-16 text-xl",
  xl: "h-24 w-24 text-3xl",
} as const;

export function Avatar({
  seed,
  label,
  avatarUri,
  size = "md",
  className = "",
}: {
  /** Stable id to derive the color from -- always the user's id, not their
   * (changeable) handle/displayName. */
  seed: string;
  /** Display name/handle shown as the initial -- first codepoint, uppercased. */
  label: string;
  avatarUri?: string | null;
  size?: keyof typeof SIZE_CLASSES;
  className?: string;
}) {
  const sizeClass = SIZE_CLASSES[size];

  if (avatarUri) {
    return (
      <img
        src={avatarUri}
        alt=""
        className={`shrink-0 rounded-full object-cover ${sizeClass} ${className}`}
      />
    );
  }

  const initial = [...label.trim()][0]?.toUpperCase() ?? "?";
  return (
    <div
      aria-hidden
      className={`flex shrink-0 items-center justify-center rounded-full font-semibold text-white ${sizeClass} ${className}`}
      style={{ backgroundColor: colorFor(seed) }}
    >
      {initial}
    </div>
  );
}
