import { type Ctx, forward } from "@/lib/proxy";

const handle = async (req: Request, { params }: Ctx) =>
  forward(req, "datasets", (await params).path);

export { handle as GET, handle as POST, handle as PATCH, handle as DELETE };
