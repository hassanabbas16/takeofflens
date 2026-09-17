"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, getPlan, isTerminal, uploadPlan, type PlanStatus } from "@/lib/api";

const ACCEPT = ".pdf,.png,.jpg,.jpeg";
const MAX_BYTES = 20 * 1024 * 1024;
const POLL_MS = 1500;

type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; filename: string }
  | { kind: "processing"; id: string; filename: string; status: PlanStatus }
  | { kind: "error"; message: string };

export default function Home() {
  const router = useRouter();
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const busy = phase.kind === "uploading" || phase.kind === "processing";

  const start = useCallback(async (file: File) => {
    // Checked here as well as server-side so the user hears about it before a 20MB upload.
    if (file.size > MAX_BYTES) {
      setPhase({
        kind: "error",
        message: `${file.name} is ${(file.size / 1024 / 1024).toFixed(1)} MB; the limit is 20 MB.`,
      });
      return;
    }
    setPhase({ kind: "uploading", filename: file.name });
    try {
      const { id, status } = await uploadPlan(file);
      setPhase({ kind: "processing", id, filename: file.name, status });
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Could not reach the API. Is the api container running?";
      setPhase({ kind: "error", message });
    }
  }, []);

  // Poll until the job reaches a terminal state, then hand over to the viewer.
  useEffect(() => {
    if (phase.kind !== "processing" || isTerminal(phase.status)) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const plan = await getPlan(phase.id, controller.signal);
        if (plan.status === "done") {
          router.push(`/plans/${plan.id}`);
          return;
        }
        if (plan.status === "failed") {
          setPhase({ kind: "error", message: plan.error ?? "Processing failed." });
          return;
        }
        setPhase({ ...phase, status: plan.status });
      } catch (err) {
        if (controller.signal.aborted) return;
        setPhase({
          kind: "error",
          message: err instanceof ApiError ? err.message : "Lost contact with the API.",
        });
      }
    }, POLL_MS);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [phase, router]);

  const onDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (busy) return;
    const file = event.dataTransfer.files[0];
    if (file) void start(file);
  };

  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">TakeoffLens</h1>
      <p className="mt-3 text-neutral-600">
        Upload an architectural floor plan to extract rooms, labels and areas.
      </p>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!busy) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`mt-8 rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors ${
          dragging ? "border-sky-400 bg-sky-50" : "border-neutral-300 bg-white"
        } ${busy ? "opacity-60" : ""}`}
      >
        <p className="text-sm text-neutral-600">
          Drag a plan here, or{" "}
          <button
            type="button"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
            className="font-medium text-sky-700 underline underline-offset-2 disabled:no-underline"
          >
            choose a file
          </button>
          .
        </p>
        <p className="mt-2 text-xs text-neutral-400">PDF, PNG or JPG, up to 20 MB.</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void start(file);
            e.target.value = "";
          }}
        />
      </div>

      {phase.kind === "uploading" && (
        <Status tone="busy">Uploading {phase.filename}…</Status>
      )}

      {phase.kind === "processing" && (
        <Status tone="busy">
          Processing {phase.filename} — {phase.status}. Rendering the page, running OCR and
          extracting rooms; this takes a minute or two on CPU.
        </Status>
      )}

      {phase.kind === "error" && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          <p>{phase.message}</p>
          <button
            type="button"
            onClick={() => setPhase({ kind: "idle" })}
            className="mt-2 font-medium underline underline-offset-2"
          >
            Try another file
          </button>
        </div>
      )}
    </main>
  );
}

function Status({ tone, children }: { tone: "busy"; children: React.ReactNode }) {
  return (
    <div className="mt-6 flex items-start gap-3 rounded-lg border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-700">
      {tone === "busy" && (
        <span className="mt-0.5 h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-neutral-300 border-t-sky-600" />
      )}
      <p>{children}</p>
    </div>
  );
}
