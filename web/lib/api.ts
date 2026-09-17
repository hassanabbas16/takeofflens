/**
 * Typed client for the TakeoffLens API.
 *
 * The shapes here mirror `api/app/schemas.py` exactly. They are hand-written rather than
 * generated because there are only four of them; if the schema grows, generate from the
 * OpenAPI document instead of widening these by hand.
 */

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** [x1, y1, x2, y2] in *page image pixel* space, not screen space. */
export type BBox = [number, number, number, number];

export interface Token {
  id: number;
  text: string;
  confidence: number;
  bbox: number[];
}

export interface Room {
  id: string;
  name: string | null;
  room_type: string;
  width_m: number | null;
  length_m: number | null;
  area_m2: number | null;
  source: string;
  raw_text: string | null;
  bbox: number[] | null;
  confidence: number | null;
  grounded: boolean;
}

export interface Page {
  id: string;
  page_number: number;
  width: number;
  height: number;
  tokens: Token[];
  rooms: Room[];
}

export type PlanStatus = "pending" | "processing" | "done" | "failed";

export interface Plan {
  id: string;
  filename: string;
  status: PlanStatus;
  error: string | null;
  created_at: string;
  pages: Page[];
}

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Pull FastAPI's `detail` out of an error body so the user sees the real reason. */
async function readError(response: Response): Promise<never> {
  let detail = response.statusText;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const value = (body as { detail: unknown }).detail;
      if (typeof value === "string") detail = value;
    }
  } catch {
    // Body was not JSON. The status text is the best we have.
  }
  throw new ApiError(response.status, detail);
}

export async function uploadPlan(file: File): Promise<{ id: string; status: PlanStatus }> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_URL}/plans`, { method: "POST", body: form });
  if (!response.ok) await readError(response);
  return (await response.json()) as { id: string; status: PlanStatus };
}

export async function getPlan(id: string, signal?: AbortSignal): Promise<Plan> {
  const response = await fetch(`${API_URL}/plans/${id}`, { signal, cache: "no-store" });
  if (!response.ok) await readError(response);
  return (await response.json()) as Plan;
}

export function pageImageUrl(planId: string, pageNumber: number): string {
  return `${API_URL}/plans/${planId}/pages/${pageNumber}/image`;
}

export function exportUrl(planId: string, format: "csv" | "json", source: string | null): string {
  const params = new URLSearchParams({ format });
  if (source) params.set("source", source);
  return `${API_URL}/plans/${planId}/export?${params.toString()}`;
}

/** A plan is still being worked on while it is in one of these states. */
export function isTerminal(status: PlanStatus): boolean {
  return status === "done" || status === "failed";
}
