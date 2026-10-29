"use client";

import { useState } from "react";

export interface Series {
  name: string;
  color: string;
  dashed?: boolean;
  points: { x: number; y: number }[];
}

interface Props {
  series: Series[];
  forecast?: { x: number; y: number; error: number; color: string; label: string };
  yLabel: string;
  format: (v: number) => string;
  zeroLine?: boolean;
  height?: number;
}

// Small single-axis line chart: 2px lines, 8px markers, recessive grid, direct end labels, legend for 2+ series,
// and a crosshair tooltip on hover.
export default function LineChart({ series, forecast, yLabel, format, zeroLine, height = 220 }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 560, H = height, L = 52, R = 96, T = 14, B = 30;
  const xs = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.x)).concat(forecast ? [forecast.x] : []))).sort((a, b) => a - b);
  const ys = series.flatMap((s) => s.points.map((p) => p.y))
    .concat(forecast ? [forecast.y - forecast.error, forecast.y + forecast.error] : [])
    .concat(zeroLine ? [0] : []);
  if (!xs.length || !ys.length) return null;
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = (yMax - yMin || 1) * 0.12;
  yMin -= pad; yMax += pad;
  const sx = (x: number) => L + ((x - xs[0]) / Math.max(1, xs[xs.length - 1] - xs[0])) * (W - L - R);
  const sy = (y: number) => T + (1 - (y - yMin) / (yMax - yMin)) * (H - T - B);
  const ticks = Array.from({ length: 5 }, (_, i) => yMin + ((i + 0.5) * (yMax - yMin)) / 5);
  const hx = hover !== null ? xs[hover] : null;

  return (
    <div>
      {series.length > 1 && (
        <div className="mb-2 flex flex-wrap gap-4 text-xs text-slate-400">
          {series.map((s) => (
            <span key={s.name} className="inline-flex items-center gap-1.5">
              <svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? "4 3" : undefined} /></svg>
              {s.name}
            </span>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={yLabel}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const px = ((e.clientX - rect.left) / rect.width) * W;
          let best = 0;
          xs.forEach((x, i) => { if (Math.abs(sx(x) - px) < Math.abs(sx(xs[best]) - px)) best = i; });
          setHover(best);
        }}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="rgba(148,163,184,0.12)" />
            <text x={L - 6} y={sy(t) + 3} textAnchor="end" fontSize="10" fill="#94a3b8">{format(t)}</text>
          </g>
        ))}
        {zeroLine && yMin < 0 && yMax > 0 && <line x1={L} x2={W - R} y1={sy(0)} y2={sy(0)} stroke="rgba(148,163,184,0.45)" />}
        {xs.map((x) => (
          <text key={x} x={sx(x)} y={H - 10} textAnchor="middle" fontSize="10" fill="#94a3b8">{`'${String(x).slice(2)}`}</text>
        ))}
        {hx !== null && <line x1={sx(hx)} x2={sx(hx)} y1={T} y2={H - B} stroke="rgba(148,163,184,0.35)" />}
        {series.map((s) => {
          const pts = [...s.points].sort((a, b) => a.x - b.x);
          const last = pts[pts.length - 1];
          return (
            <g key={s.name}>
              <polyline fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                strokeDasharray={s.dashed ? "5 4" : undefined} points={pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
              {!s.dashed && pts.map((p) => (
                <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={hx === p.x ? 5 : 3.5} fill={s.color} stroke="#162033" strokeWidth="2" />
              ))}
              {last && series.length > 1 && <text x={sx(last.x) + 8} y={sy(last.y) + 3} fontSize="10" fill="#cbd5e1">{s.name}</text>}
            </g>
          );
        })}
        {forecast && (
          <g>
            <line x1={sx(forecast.x)} x2={sx(forecast.x)} y1={sy(forecast.y - forecast.error)} y2={sy(forecast.y + forecast.error)}
              stroke={forecast.color} strokeWidth="2" strokeOpacity="0.6" strokeLinecap="round" />
            <circle cx={sx(forecast.x)} cy={sy(forecast.y)} r="5" fill="#162033" stroke={forecast.color} strokeWidth="2" />
            <text x={sx(forecast.x) + 9} y={sy(forecast.y) + 3} fontSize="10" fill="#cbd5e1">{forecast.label}</text>
          </g>
        )}
        {hx !== null && (() => {
          const rows = series.map((s) => ({ s, p: s.points.find((p) => p.x === hx) })).filter((r) => r.p);
          const f = forecast && forecast.x === hx ? forecast : null;
          if (!rows.length && !f) return null;
          const bx = Math.min(sx(hx) + 10, W - 170);
          const lines = rows.length + (f ? 1 : 0);
          return (
            <g pointerEvents="none">
              <rect x={bx} y={T} width="160" height={20 + 15 * lines} rx="6" fill="#0f1729" stroke="rgba(255,255,255,0.15)" />
              <text x={bx + 10} y={T + 15} fontSize="10" fill="#f1f5f9" fontWeight="600">FY{hx}</text>
              {rows.map((r, i) => (
                <text key={r.s.name} x={bx + 10} y={T + 30 + 15 * i} fontSize="10" fill="#cbd5e1">
                  <tspan fill={r.s.color}>■ </tspan>{r.s.name}: {format(r.p!.y)}
                </text>
              ))}
              {f && <text x={bx + 10} y={T + 30 + 15 * rows.length} fontSize="10" fill="#cbd5e1">Forecast: {format(f.y)} ± {format(f.error)}</text>}
            </g>
          );
        })()}
      </svg>
    </div>
  );
}
