import { proxyToPython } from "@/lib/proxy-python";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: Request) {
  return proxyToPython(request, "/api/py-income", { method: "GET" });
}

export async function POST(request: Request) {
  const body = await request.text();
  return proxyToPython(request, "/api/py-income", {
    method: "POST",
    body: body || "{}",
  });
}
