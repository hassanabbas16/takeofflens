"use client";

import type { Room } from "@/lib/api";

interface Props {
  rooms: Room[];
  highlightedRoomId: string | null;
  onHoverRoom: (id: string | null) => void;
}

function formatMetres(value: number | null): string {
  // A null here is a real measurement, not a gap: these plans print a single area under the
  // room label and no width/length pair, and the pipeline never derives one from the other.
  return value === null ? "—" : value.toFixed(2);
}

function formatArea(value: number | null): string {
  return value === null ? "—" : value.toFixed(1);
}

export default function RoomsTable({ rooms, highlightedRoomId, onHoverRoom }: Props) {
  if (rooms.length === 0) {
    return (
      <p className="rounded-lg border border-neutral-200 bg-white px-4 py-6 text-sm text-neutral-500">
        No rooms extracted for this approach.
      </p>
    );
  }

  const withArea = rooms.filter((r) => r.area_m2 !== null);
  const totalArea = withArea.reduce((sum, r) => sum + (r.area_m2 ?? 0), 0);

  return (
    <div className="overflow-hidden rounded-lg border border-neutral-200 bg-white">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-neutral-200 bg-neutral-50 text-left text-xs uppercase tracking-wide text-neutral-500">
            <th scope="col" className="px-3 py-2 font-medium">Label</th>
            <th scope="col" className="px-3 py-2 font-medium">Type</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">W (m)</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">L (m)</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Area (m²)</th>
          </tr>
        </thead>
        <tbody>
          {rooms.map((room) => {
            const active = room.id === highlightedRoomId;
            return (
              <tr
                key={room.id}
                onMouseEnter={() => onHoverRoom(room.id)}
                onMouseLeave={() => onHoverRoom(null)}
                className={`border-b border-neutral-100 last:border-0 transition-colors ${
                  active ? "bg-amber-50" : "hover:bg-neutral-50"
                }`}
              >
                <td className="px-3 py-2">
                  <span className="font-medium text-neutral-900">{room.name ?? "—"}</span>
                  {!room.grounded && (
                    <span
                      className="ml-2 rounded bg-rose-100 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-rose-700"
                      title="This room could not be tied back to an OCR token on the page. It is kept and shown rather than dropped, so it stays countable."
                    >
                      ungrounded
                    </span>
                  )}
                  {room.raw_text && room.raw_text !== room.name && (
                    <span className="mt-0.5 block font-mono text-[11px] text-neutral-400">
                      {room.raw_text}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-neutral-600">{room.room_type}</td>
                <td className="px-3 py-2 text-right tabular-nums text-neutral-600">
                  {formatMetres(room.width_m)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-neutral-600">
                  {formatMetres(room.length_m)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums font-medium text-neutral-900">
                  {formatArea(room.area_m2)}
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="bg-neutral-50 text-xs text-neutral-600">
            <td className="px-3 py-2" colSpan={4}>
              {rooms.length} room{rooms.length === 1 ? "" : "s"}, {withArea.length} with an area
            </td>
            <td className="px-3 py-2 text-right tabular-nums font-medium text-neutral-900">
              {withArea.length > 0 ? totalArea.toFixed(1) : "—"}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
