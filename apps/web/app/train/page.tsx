import { listDatasets } from "@/lib/api";
import { TrainPanel } from "./TrainPanel";

export const dynamic = "force-dynamic";

export default async function TrainPage() {
  return <TrainPanel initial={await listDatasets()} />;
}
