"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import PlanCanvas from "@/components/PlanCanvas";
import RoomsTable from "@/components/RoomsTable";
import {
  ApiError,
  exportUrl,
  getPlan,
  isTerminal,
  pageImageUrl,
  type Plan,
} from "@/lib/api";

const POLL_MS = 1500;

/** Friendly names for the extraction approaches, in the order they are worth comparing. */
const SOURCE_LABELS: Record<string, string> = {
  rules: "Rules",
  "ocr+llm": "OCR + LLM",
  vlm: "VLM",
  hybrid: "Hybrid",
  detector: "Detector",
};
const SOURCE_ORDER = ["rules", "ocr+llm", "vlm", "hybrid", "detector"];

export default function PlanViewer() {
  const params = useParams<{ id: string }>();
  const id = params.id;

  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [source, setSource] = useState<string | null>(null);
  const [showTokens, setShowTokens] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);

  // Load, and keep polling while the job is still running so a link opened mid-processing
  // fills itself in rather than showing an empty plan.
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      try {
        const next = await getPlan(id, controller.signal);
        if (cancelled) return;
        setPlan(next);
        setError(null);
        if (!isTerminal(next.status)) timer = setTimeout(tick, POLL_MS);
      } catch (err) {
        if (cancelled || controller.signal.aborted) return;
        setError(err instanceof ApiError ? err.message : "Could not reach the API.");
      }
    };
    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer!);
    };
  }, [id]);

  const page = plan?.pages[pageIndex];

  /** Which approaches actually produced rooms on this page. */
  const sources = useMemo(() => {
    if (!page) return [];
    const present = new Set(page.rooms.map((r) => r.source));
    const known = SOURCE_ORDER.filter((s) => present.has(s));
    const extra = [...present].filter((s) => !SOURCE_ORDER.includes(s)).sort();
    return [...known, ...extra];
  }, [page]);

  // Default to the first approach present rather than a hardcoded one, so the page is
  // correct whichever approaches the pipeline was configured to run.
  useEffect(() => {
    if (source === null && sources.length > 0) setSource(sources[0] ?? null);
  }, [sources, source]);

  const rooms = useMemo(
    () => (page ? page.rooms.filter((r) => r.source === source) : []),
    [page, source],
  );

  if (error) {
    return (
      <Shell>
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </div>
      </Shell>
    );
  }

  if (!plan) {
    return (
      <Shell>
        <p className="text-sm text-neutral-500">Loading plan…</p>
      </Shell>
    );
  }

  if (plan.status === "failed") {
    return (
      <Shell filename={plan.filename}>
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          <p className="font-medium">Processing failed.</p>
          <p className="mt-1">{plan.error ?? "No reason was recorded."}</p>
        </div>
      </Shell>
    );
  }

  if (!isTerminal(plan.status) || !page) {
    return (
      <Shell filename={plan.filename}>
        <div className="flex items-center gap-3 text-sm text-neutral-600">
          <span className="h-3 w-3 animate-spin rounded-full border-2 border-neutral-300 border-t-sky-600" />
          <p>Processing — {plan.status}. This page updates itself when it finishes.</p>
        </div>
      </Shell>
    );
  }

  return (
    <Shell filename={plan.filename}>
      <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-3">
        {sources.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-neutral-500">Approach</span>
            <div className="flex overflow-hidden rounded-md border border-neutral-300">
              {sources.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSource(s)}
                  className={`px-3 py-1.5 text-sm transition-colors ${
                    s === source
                      ? "bg-neutral-900 text-white"
                      : "bg-white text-neutral-700 hover:bg-neutral-50"
                  }`}
                >
                  {SOURCE_LABELS[s] ?? s}
                </button>
              ))}
            </div>
          </div>
        )}

        <label className="flex cursor-pointer items-center gap-2 text-sm text-neutral-700">
          <input
            type="checkbox"
            checked={showTokens}
            onChange={(e) => setShowTokens(e.target.checked)}
            className="h-4 w-4 rounded border-neutral-300"
          />
          OCR boxes ({page.tokens.length})
        </label>

        {plan.pages.length > 1 && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-xs uppercase tracking-wide text-neutral-500">Page</span>
            <select
              value={pageIndex}
              onChange={(e) => setPageIndex(Number(e.target.value))}
              className="rounded-md border border-neutral-300 bg-white px-2 py-1"
            >
              {plan.pages.map((p, i) => (
                <option key={p.id} value={i}>
                  {p.page_number}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="ml-auto flex items-center gap-2">
          <a
            href={exportUrl(plan.id, "csv", source)}
            className="rounded-md border border-neutral-300 bg-white px-3 py-1.5 text-sm text-neutral-700 hover:bg-neutral-50"
          >
            Export CSV
          </a>
          <a
            href={exportUrl(plan.id, "json", source)}
            className="rounded-md border border-neutral-300 bg-white px-3 py-1.5 text-sm text-neutral-700 hover:bg-neutral-50"
          >
            Export JSON
          </a>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_28rem]">
        <PlanCanvas
          imageUrl={pageImageUrl(plan.id, page.page_number)}
          page={page}
          rooms={rooms}
          showTokens={showTokens}
          highlightedRoomId={hovered}
          onHoverRoom={setHovered}
        />
        <div>
          <RoomsTable rooms={rooms} highlightedRoomId={hovered} onHoverRoom={setHovered} />
          <p className="mt-3 text-xs leading-relaxed text-neutral-500">
            A dash means the value was not printed on the page. Width and length are only
            filled in when the drawing prints a <span className="font-mono">w × l</span> pair;
            they are never derived from an area.
          </p>
        </div>
      </div>
    </Shell>
  );
}

function Shell({ filename, children }: { filename?: string; children: React.ReactNode }) {
  return (
    <main className="mx-auto max-w-[110rem] px-6 py-10">
      <div className="mb-6">
        <Link href="/" className="text-sm text-sky-700 underline underline-offset-2">
          ← Upload another plan
        </Link>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight">
          {filename ?? "Plan"}
        </h1>
      </div>
      {children}
    </main>
  );
}
