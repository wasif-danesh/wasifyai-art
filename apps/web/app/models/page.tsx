import { listFamilies, listModels } from "@/lib/api";
import { ModelsPanel } from "./ModelsPanel";

export const dynamic = "force-dynamic";

export default async function ModelsPage() {
  const [models, families] = await Promise.all([listModels(), listFamilies()]);
  return <ModelsPanel initialModels={models} families={families} />;
}
