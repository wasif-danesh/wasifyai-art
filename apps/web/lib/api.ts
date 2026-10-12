// Server-side only: the browser never calls the API directly (BFF rule). The base URL comes from the
// environment, never a hard-coded host.
import type {
  CreationSummary,
  FamilyInfo,
  ModelSummary,
  ProviderInfo,
} from "@wasifyai-art/contracts";

export const apiBase = () => process.env.API_BASE_URL ?? "http://localhost:8010";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${apiBase()}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`API ${path} answered ${r.status}`);
  return (await r.json()) as T;
}

export async function listProviders(): Promise<ProviderInfo[]> {
  return (await get<{ providers: ProviderInfo[] }>("/v1/providers")).providers;
}

export async function listModels(): Promise<ModelSummary[]> {
  return (await get<{ models: ModelSummary[] }>("/v1/models")).models;
}

export async function listFamilies(): Promise<FamilyInfo[]> {
  return (await get<{ families: FamilyInfo[] }>("/v1/families")).families;
}

export async function listCreations(): Promise<CreationSummary[]> {
  return (await get<{ creations: CreationSummary[] }>("/v1/creations")).creations;
}

export async function getCreation(id: string): Promise<CreationSummary | null> {
  const r = await fetch(`${apiBase()}/v1/creations/${encodeURIComponent(id)}`, {
    cache: "no-store",
  });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`API creation answered ${r.status}`);
  return (await r.json()) as CreationSummary;
}
