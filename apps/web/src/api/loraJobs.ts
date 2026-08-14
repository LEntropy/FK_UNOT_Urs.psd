import { api } from "./client";
import { useAuthStore } from "../store/auth";

export interface LoraGenerationJob {
  id: string;
  status: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
  errorMessage: string | null;
}

export const createLoraJob = (sourceArtworkId: string) =>
  api.post<{ id: string }>("/lora-jobs", { sourceArtworkId });

export const getLoraJob = (id: string) => api.get<LoraGenerationJob>(`/lora-jobs/${id}`);

const BASE_URL = import.meta.env.VITE_API_GATEWAY_URL ?? "http://localhost:4000";

/** Downloads the trained .safetensors and triggers a browser save -- not
 * a plain <a href>, since the download route needs the Authorization
 * header (see client.ts's rawFetch for why every other request already
 * gets this from useAuthStore automatically; a bare <a> tag can't). */
export async function downloadLoraJob(id: string): Promise<void> {
  const accessToken = useAuthStore.getState().accessToken;
  const res = await fetch(`${BASE_URL}/lora-jobs/${id}/download`, {
    headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
  });
  if (!res.ok) throw new Error(`download failed: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${id}.safetensors`;
  a.click();
  URL.revokeObjectURL(url);
}
