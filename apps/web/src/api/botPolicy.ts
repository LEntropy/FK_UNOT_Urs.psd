import { api } from "./client";

export type BotAction = "ALLOW" | "BLOCK" | "LOG_ONLY";

export const BOT_GROUPS = [
  "search_engine",
  "ai_training",
  "ai_search",
  "ai_assistant",
  "social_preview",
  "seo",
  "unknown_bot",
] as const;
export type BotGroup = (typeof BOT_GROUPS)[number];

export const BOT_GROUP_LABEL: Record<BotGroup, string> = {
  search_engine: "검색엔진 (Google/Bing 등)",
  ai_training: "AI 학습용 크롤러 (GPTBot 등)",
  ai_search: "AI 검색 크롤러",
  ai_assistant: "AI 어시스턴트 봇",
  social_preview: "SNS 미리보기 (트위터 등)",
  seo: "SEO 분석봇",
  unknown_bot: "미분류 봇",
};

export interface ArtworkBotPolicy {
  defaultAction: BotAction;
  groupPolicies: Partial<Record<BotGroup, BotAction>>;
  botOverrides: Record<string, BotAction>;
}

export interface BotAccessLogEntry {
  id: number;
  artworkId: string;
  botName: string;
  botGroup: string;
  action: BotAction;
  clientIp: string;
  userAgent: string;
  responseStatus: number;
  timestamp: string;
}

export const getBotPolicy = (artworkId: string) => api.get<ArtworkBotPolicy>(`/artworks/${artworkId}/bot-policy`);

export const setBotPolicy = (artworkId: string, policy: ArtworkBotPolicy) =>
  api.put<ArtworkBotPolicy>(`/artworks/${artworkId}/bot-policy`, policy);

export const getBotAccessLogs = (artworkId: string, limit = 50) =>
  api.get<BotAccessLogEntry[]>(`/artworks/${artworkId}/bot-access-logs?limit=${limit}`);
