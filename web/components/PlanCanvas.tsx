"use client";

import { useState } from "react";
import type { Page, Room, Token } from "@/lib/api";

interface Props {
  imageUrl: string;
  page: Page;
  rooms: Room[];
  showTokens: boolean;
  /** Room id hovered in the table, so the canvas can highlight the matching box. */
  highlightedRoomId: string | null;
  onHoverRoom: (id: string | null) => void;
}

/**
 * The page image with OCR/room boxes drawn over it.
 *
 * Boxes are in *image pixel* space, which is not screen space: the image is scaled to fit
 * the column and the plan is far larger than any viewport. Rather than measuring the
 * rendered <img> and converting on every resize, the overlay is an SVG with
 * `viewBox="0 0 width height"` stacked on the image at the same size. The browser then
 * applies exactly the same scale to both, so a box drawn at the token's raw pixel
 * coordinates lands on the token at every zoom level and window size, with no JS.
 */
export default function PlanCanvas({
  imageUrl,
  page,
  rooms,
  showTokens,
  highlightedRoomId,
  onHoverRoom,
}: Props) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);

  const roomBoxes = rooms.filter(
    (room): room is Room & { bbox: number[] } => room.bbox !== null && room.bbox.length === 4,
  );

  return (
    <div className="relative overflow-hidden rounded-lg border border-neutral-200 bg-white">
      {!loaded && !failed && (
        <div className="flex h-96 items-center justify-center text-sm text-neutral-500">
          Loading page image…
        </div>
      )}
      {failed && (
        <div className="flex h-96 items-center justify-center px-6 text-center text-sm text-red-600">
          Could not load the page image.
        </div>
      )}

      <div className="relative" style={{ display: failed ? "none" : undefined }}>
        {/* eslint-disable-next-line @next/next/no-img-element -- the API streams this; next/image would need a remote loader for no benefit here. */}
        <img
          src={imageUrl}
          alt={`Page ${page.page_number}`}
          className="block w-full select-none"
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
        />

        {loaded && (
          <svg
            viewBox={`0 0 ${page.width} ${page.height}`}
            preserveAspectRatio="none"
            className="pointer-events-none absolute inset-0 h-full w-full"
            aria-hidden="true"
          >
            {showTokens &&
              page.tokens.map((token: Token) => {
                const [x1, y1, x2, y2] = token.bbox;
                if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) {
                  return null;
                }
                return (
                  <rect
                    key={`t-${token.id}`}
                    x={x1}
                    y={y1}
                    width={Math.max(1, x2 - x1)}
                    height={Math.max(1, y2 - y1)}
                    className="fill-sky-400/10 stroke-sky-500/70"
                    strokeWidth={2}
                  />
                );
              })}

            {roomBoxes.map((room) => {
              const [x1, y1, x2, y2] = room.bbox;
              if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) {
                return null;
              }
              const active = room.id === highlightedRoomId;
              return (
                <g key={`r-${room.id}`}>
                  <rect
                    x={x1}
                    y={y1}
                    width={Math.max(1, x2 - x1)}
                    height={Math.max(1, y2 - y1)}
                    className={
                      active
                        ? "fill-amber-400/30 stroke-amber-500"
                        : room.grounded
                          ? "fill-emerald-400/10 stroke-emerald-600/80"
                          : // Ungrounded rooms are kept and shown, not hidden: a claim we
                            // could not tie back to a token is exactly what a reviewer
                            // needs to see.
                            "fill-rose-400/10 stroke-rose-500/80"
                    }
                    strokeWidth={active ? 5 : 3}
                  />
                </g>
              );
            })}
          </svg>
        )}

        {/* A parallel, interactive layer: the SVG above is pointer-events-none so the
            boxes never block the image, and hover is handled by these absolutely
            positioned targets expressed in percentages of the page. */}
        {loaded &&
          roomBoxes.map((room) => {
            const [x1, y1, x2, y2] = room.bbox;
            if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) {
              return null;
            }
            return (
              <button
                key={`h-${room.id}`}
                type="button"
                className="absolute cursor-pointer"
                style={{
                  left: `${(x1 / page.width) * 100}%`,
                  top: `${(y1 / page.height) * 100}%`,
                  width: `${((x2 - x1) / page.width) * 100}%`,
                  height: `${((y2 - y1) / page.height) * 100}%`,
                }}
                onMouseEnter={() => onHoverRoom(room.id)}
                onMouseLeave={() => onHoverRoom(null)}
                onFocus={() => onHoverRoom(room.id)}
                onBlur={() => onHoverRoom(null)}
                aria-label={`${room.name ?? room.room_type}${
                  room.area_m2 === null ? "" : `, ${room.area_m2} square metres`
                }`}
              />
            );
          })}
      </div>
    </div>
  );
}
