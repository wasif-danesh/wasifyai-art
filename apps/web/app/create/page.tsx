import { getCreation, listModels, listProviders } from "@/lib/api";
import { CreateForm, type Initial } from "./CreateForm";

export const dynamic = "force-dynamic";

type Search = Promise<{ remix?: string; edit?: string }>;

export default async function CreatePage({ searchParams }: { searchParams: Search }) {
  const q = await searchParams;
  const [providers, models] = await Promise.all([listProviders(), listModels()]);
  const bases = models
    .filter((m) => m.kind === "checkpoint")
    .flatMap((m) =>
      m.versions.map((v) => ({ id: v.id, name: m.name, status: v.status, family: v.family })),
    )
    .filter((v) => v.status === "ready");

  const loras = models
    .filter((m) => m.kind === "lora")
    .flatMap((m) =>
      m.versions
        .filter((v) => v.status === "ready")
        .map((v) => ({ id: v.id, name: m.name, family: v.family, triggerWords: v.trigger_words })),
    );

  let initial: Initial | undefined;
  const fromId = q.edit ?? q.remix;
  if (fromId) {
    const c = await getCreation(fromId);
    if (c) {
      const size = c.params.size;
      initial = {
        mode: q.edit ? "edit" : "text",
        prompt: c.prompt,
        versionId: c.models[0]?.version_id ?? null,
        modelName: c.models[0]?.name ?? "the original model",
        size: typeof size === "string" ? size : "square",
        seed: typeof c.params.seed === "number" ? c.params.seed : null,
        sourceId: c.id,
        loras: c.models
          .filter((m) => m.role === "lora" && m.version_id)
          .map((m) => ({ versionId: m.version_id as string, name: m.name, weight: m.weight ?? 1 })),
      };
    }
  }
  return <CreateForm providers={providers} bases={bases} loras={loras} initial={initial} />;
}
