import { NextResponse } from "next/server";

/** Base URL for same-deployment Python serverless functions. */
export function getDeploymentOrigin(request: Request): string {
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`;
  }
  return new URL(request.url).origin;
}

export async function proxyToPython(
  request: Request,
  pythonPath: string,
  init?: RequestInit
): Promise<NextResponse> {
  const origin = getDeploymentOrigin(request);
  const url = `${origin}${pythonPath}`;

  const headers = new Headers(init?.headers);
  if (!headers.has("Content-Type") && init?.body) {
    headers.set("Content-Type", "application/json");
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, { ...init, headers });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Upstream request failed";
    return NextResponse.json(
      {
        error: `ML API unreachable at ${pythonPath}. Run \`vercel dev\` locally or check deployment logs.`,
        detail: message,
      },
      { status: 503 }
    );
  }

  const text = await upstream.text();

  if (text.trimStart().startsWith("<")) {
    return NextResponse.json(
      {
        error:
          "ML API returned HTML instead of JSON. Use `vercel dev` for local testing, or confirm Python functions are deployed (api/py-shortage.py).",
        status: upstream.status,
      },
      { status: 503 }
    );
  }

  try {
    const data = JSON.parse(text);
    return NextResponse.json(data, { status: upstream.status });
  } catch {
    return NextResponse.json(
      { error: "ML API returned non-JSON response.", body: text.slice(0, 200) },
      { status: 502 }
    );
  }
}
