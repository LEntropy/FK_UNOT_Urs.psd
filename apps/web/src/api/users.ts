import { api } from "./client";

export interface PublicProfile {
  id: string;
  handle: string;
  displayName: string | null;
  avatarUri: string | null;
  bio: string | null;
  role: string;
}

export interface MeProfile extends PublicProfile {
  email: string;
  walletAddress: string;
  createdAt: string;
  status: string;
}

export const getMe = () => api.get<MeProfile>("/me");
export const updateMe = (patch: { displayName?: string; bio?: string }) => api.patch<MeProfile>("/me", patch);

export const getUser = (id: string) => api.get<PublicProfile>(`/users/${id}`);
export const getUsersBatch = (ids: string[]): Promise<PublicProfile[]> =>
  ids.length === 0 ? Promise.resolve([]) : api.get<PublicProfile[]>(`/users?ids=${ids.map(encodeURIComponent).join(",")}`);
