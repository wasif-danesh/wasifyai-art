import { getDataset } from "@/lib/api";
import { notFound } from "next/navigation";
import { DatasetEditor } from "./DatasetEditor";

export const dynamic = "force-dynamic";

export default async function DatasetPage({ params }: { params: Promise<{ id: string }> }) {
  const dataset = await getDataset((await params).id);
  if (!dataset) notFound();
  return <DatasetEditor initial={dataset} />;
}
