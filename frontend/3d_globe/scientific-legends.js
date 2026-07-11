/**
 * Scientific display ranges, colormaps, and floating legend markup for Mars GeoTIFF overlays.
 * Fixed value domains (not per-tile min–max) so the globe tint matches published literature scales.
 */

/** @typedef {{ t: number; rgb: [number, number, number] }} ColorStop */
/** @typedef {{ value: number; num: string; qual: string }} LegendTick */

/** Landing suitability ML overlay (0–100%). */
export const SUITABILITY_LEGEND = {
  id: "suitability",
  title: "Landing suitability",
  unit: "%",
  displayMin: 0,
  displayMax: 100,
  colorStops: [
    { t: 0, rgb: [24, 52, 138] },
    { t: 0.35, rgb: [56, 130, 210] },
    { t: 0.5, rgb: [255, 208, 68] },
    { t: 0.75, rgb: [255, 118, 42] },
    { t: 1, rgb: [188, 26, 48] },
  ],
  ticks: [
    { value: 0, num: "0%", qual: "Poor" },
    { value: 25, num: "25%", qual: "Fair" },
    { value: 50, num: "50%", qual: "Moderate" },
    { value: 70, num: "70%", qual: "Good" },
    { value: 100, num: "100%", qual: "Excellent" },
  ],
  note: "No site on Mars currently exceeds ~70% under VANGUARD engineering safety criteria.",
  citation: "Golombek et al. 2012; NASA Mars 2020 landing constraints",
  accentRgb: [255, 180, 80],
};

/** Globe surface layer keys → scientific legend config. */
export const SCIENTIFIC_LAYER_LEGENDS = {
  slope: {
    title: "Slope",
    unit: "°",
    displayMin: 0,
    displayMax: 30,
    colorStops: [
      { t: 0, rgb: [46, 125, 50] },
      { t: 0.07, rgb: [102, 187, 106] },
      { t: 0.17, rgb: [174, 213, 129] },
      { t: 0.5, rgb: [255, 183, 77] },
      { t: 1, rgb: [198, 40, 40] },
    ],
    ticks: [
      { value: 0, num: "0–2°", qual: "Extremely flat" },
      { value: 5, num: "5°", qual: "Rover-safe" },
      { value: 15, num: "15°", qual: "Moderate" },
      { value: 30, num: "30°", qual: "Steep / EDL cutoff" },
    ],
    footnote: ">30°: extreme terrain — NASA hard EDL cutoff.",
    citation: "Golombek et al. 2012; Anderson et al. 2003 (MER)",
    accentRgb: [102, 187, 106],
  },
  thermalInertiaObs: {
    title: "Thermal inertia",
    unit: "TIU",
    displayMin: 0,
    displayMax: 800,
    colorStops: [
      { t: 0, rgb: [30, 80, 165] },
      { t: 0.125, rgb: [70, 130, 210] },
      { t: 0.5, rgb: [230, 140, 55] },
      { t: 0.75, rgb: [240, 200, 150] },
      { t: 1, rgb: [252, 250, 248] },
    ],
    ticks: [
      { value: 0, num: "<100", qual: "Fine dust" },
      { value: 100, num: "100", qual: "MSL minimum" },
      { value: 200, num: "200", qual: "Sandy mix" },
      { value: 400, num: "400", qual: "Rock mix" },
      { value: 800, num: "800+", qual: "Bedrock" },
    ],
    footnote: "Jezero typical ~200–300 TIU (Ahern et al. 2021).",
    citation: "Putzig & Mellon 2007; Golombek et al. 2012 (MSL)",
    accentRgb: [230, 140, 55],
  },
  temperature: {
    title: "Surface temperature",
    unit: "°C",
    displayMin: -143,
    displayMax: 27,
    colorStops: [
      { t: 0, rgb: [12, 28, 110] },
      { t: 0.28, rgb: [40, 90, 175] },
      { t: 0.55, rgb: [120, 150, 200] },
      { t: 0.78, rgb: [210, 140, 90] },
      { t: 1, rgb: [255, 130, 50] },
    ],
    ticks: [
      { value: -143, num: "−143°C", qual: "Mars min" },
      { value: -100, num: "−100°C", qual: "Polar winter" },
      { value: -60, num: "−60°C", qual: "High latitude" },
      { value: -40, num: "−40°C", qual: "Mid-latitude" },
      { value: -20, num: "−20°C", qual: "Equatorial warm" },
      { value: 27, num: "+27°C", qual: "Warmest" },
    ],
    citation: "NASA Mars 2020 constraints; Viking / Curiosity / Perseverance",
    accentRgb: [210, 140, 90],
  },
  ferric: {
    title: "OMEGA ferric / dust",
    unit: "index",
    displayMin: 0,
    displayMax: 2,
    colorStops: [
      { t: 0, rgb: [56, 142, 72] },
      { t: 0.25, rgb: [120, 185, 100] },
      { t: 0.5, rgb: [210, 190, 90] },
      { t: 0.75, rgb: [220, 110, 70] },
      { t: 1, rgb: [175, 35, 35] },
    ],
    ticks: [
      { value: 0, num: "0.0", qual: "Low dust" },
      { value: 0.5, num: "0.5", qual: "Rocky mix" },
      { value: 0.8, num: "0.8", qual: "Dusty" },
      { value: 1.2, num: "1.2", qual: "Stability risk" },
      { value: 2, num: "2.0", qual: "Heavily coated" },
    ],
    citation: "Ody et al. 2012 (OMEGA); Golombek et al. 2012",
    accentRgb: [120, 185, 100],
  },
  elevation: {
    title: "Elevation (MOLA)",
    unit: "m",
    displayMin: -8200,
    displayMax: 21229,
    colorStops: [
      { t: 0, rgb: [24, 52, 138] },
      { t: 0.2, rgb: [40, 120, 165] },
      { t: 0.4, rgb: [90, 165, 110] },
      { t: 0.6, rgb: [200, 200, 90] },
      { t: 0.8, rgb: [230, 120, 60] },
      { t: 1, rgb: [252, 245, 240] },
    ],
    ticks: [
      { value: -8200, num: "−8.2 km", qual: "Hellas depth" },
      { value: -4000, num: "−4 km", qual: "Deep basins" },
      { value: -1000, num: "−1 km", qual: "N. lowlands" },
      { value: 0, num: "0 m", qual: "Datum" },
      { value: 3000, num: "+3 km", qual: "Highlands" },
      { value: 21229, num: "+21 km", qual: "Olympus" },
    ],
    footnote: "Mars 2020 target elevation < −0.5 km (northern lowlands).",
    citation: "Smith et al. 1999 (MOLA); NASA Mars 2020",
    accentRgb: [90, 165, 110],
  },
  grsWaterWt: {
    title: "GRS water equivalent",
    unit: "% wt",
    displayMin: 0,
    displayMax: 10,
    colorStops: [
      { t: 0, rgb: [205, 178, 145] },
      { t: 0.35, rgb: [165, 175, 195] },
      { t: 0.65, rgb: [90, 140, 200] },
      { t: 1, rgb: [25, 75, 165] },
    ],
    ticks: [
      { value: 0, num: "0%", qual: "Dry" },
      { value: 2, num: "2%", qual: "Anhydrous" },
      { value: 4, num: "4%", qual: "Low hydration" },
      { value: 6, num: "6%", qual: "Moderate" },
      { value: 10, num: "10%", qual: "High / polar" },
    ],
    note: "GRS footprint ~520 km — regional averages, not local point measurements.",
    citation: "Feldman et al. 2004; Boynton et al. 2002 (Odyssey GRS)",
    accentRgb: [90, 140, 200],
  },
  roughness: {
    title: "Roughness (0.6 km)",
    unit: "m RMS",
    displayMin: 0,
    displayMax: 500,
    colorStops: [
      { t: 0, rgb: [56, 142, 72] },
      { t: 0.2, rgb: [120, 190, 110] },
      { t: 0.5, rgb: [230, 190, 80] },
      { t: 1, rgb: [192, 48, 42] },
    ],
    ticks: [
      { value: 0, num: "0 m", qual: "Very smooth" },
      { value: 50, num: "50 m", qual: "N. lowlands" },
      { value: 100, num: "100 m", qual: "Landing-safe" },
      { value: 200, num: "200 m", qual: "Moderate" },
      { value: 500, num: "500 m", qual: "Rough terrain" },
    ],
    citation: "Neumann et al. 2003 (MOLA); JPL roughness map",
    accentRgb: [120, 190, 110],
  },
};

/** @param {string | null | undefined} layerKey */
export function getScientificLegendConfig(layerKey) {
  if (!layerKey) return null;
  return SCIENTIFIC_LAYER_LEGENDS[layerKey] ?? null;
}

/** @param {ColorStop[]} stops @param {number} t */
export function sampleColorStops(stops, t) {
  const x = Math.max(0, Math.min(1, t));
  if (x <= stops[0].t) return [...stops[0].rgb];
  if (x >= stops[stops.length - 1].t) return [...stops[stops.length - 1].rgb];
  for (let i = 0; i < stops.length - 1; i++) {
    if (x <= stops[i + 1].t) {
      const t0 = stops[i].t;
      const t1 = stops[i + 1].t;
      const u = t1 > t0 ? (x - t0) / (t1 - t0) : 0;
      const [r0, g0, b0] = stops[i].rgb;
      const [r1, g1, b1] = stops[i + 1].rgb;
      return [
        Math.round(r0 + (r1 - r0) * u),
        Math.round(g0 + (g1 - g0) * u),
        Math.round(b0 + (b1 - b0) * u),
      ];
    }
  }
  return [...stops[stops.length - 1].rgb];
}

/** @param {ColorStop[]} stops */
function stopsToGradientCss(stops, direction = "to top") {
  const parts = stops.map(({ t, rgb }) => `rgb(${rgb[0]},${rgb[1]},${rgb[2]}) ${(t * 100).toFixed(1)}%`);
  return `linear-gradient(${direction}, ${parts.join(", ")})`;
}

/** @param {{ colorStops: ColorStop[] }} config */
export function legendConfigGradientCss(config, direction = "to top") {
  return stopsToGradientCss(config.colorStops, direction);
}

/**
 * Map raw raster value to normalized display position [0, 1] using fixed scientific range.
 * @param {{ displayMin: number; displayMax: number }} config
 */
export function scientificValueToNorm(config, value) {
  const span = config.displayMax - config.displayMin || 1e-6;
  return Math.max(0, Math.min(1, (Number(value) - config.displayMin) / span));
}

/** @param {{ colorStops: ColorStop[] }} config @param {number} norm */
export function scientificNormToRgb(config, norm) {
  return sampleColorStops(config.colorStops, norm);
}

/**
 * @param {string} text
 */
export function escapeLegendHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * @param {{ displayMin: number; displayMax: number; ticks: LegendTick[] }} config
 */
function tickPositionPercent(config, value) {
  const span = config.displayMax - config.displayMin || 1e-6;
  return Math.max(0, Math.min(100, ((value - config.displayMin) / span) * 100));
}

/**
 * Horizontal tick row with edge-aligned endpoints to avoid overlap at min/max.
 * @param {{ displayMin: number; displayMax: number; ticks: LegendTick[] }} config
 */
export function buildHorizontalTicksHtml(config) {
  const ticks = config.ticks;
  const n = ticks.length;
  if (n === 0) return "";

  const positioned = ticks.map((tick, i) => ({
    tick,
    i,
    pct: tickPositionPercent(config, tick.value),
  }));

  // Hide geological qual when ticks crowd (< 9% of scale apart).
  const hideQual = new Set();
  for (let i = 1; i < positioned.length; i += 1) {
    if (positioned[i].pct - positioned[i - 1].pct < 9) {
      hideQual.add(positioned[i].i);
      if (positioned[i].pct - positioned[i - 1].pct < 5) {
        hideQual.add(positioned[i - 1].i);
      }
    }
  }

  return positioned
    .map(({ tick, i, pct }) => {
      let alignClass = "sci-legend__tick-h--mid";
      if (pct <= 8 || i === 0) alignClass = "sci-legend__tick-h--start";
      else if (pct >= 92 || i === n - 1) alignClass = "sci-legend__tick-h--end";
      const qual =
        tick.qual != null && tick.qual !== "" && !hideQual.has(i)
          ? `<span class="sci-legend__tick-qual">${escapeLegendHtml(tick.qual)}</span>`
          : "";
      return `<div class="sci-legend__tick-h ${alignClass}" style="left:${pct.toFixed(2)}%">
        <span class="sci-legend__tick-nub" aria-hidden="true"></span>
        <span class="sci-legend__tick-val">${escapeLegendHtml(tick.num)}</span>
        ${qual}
      </div>`;
    })
    .join("");
}

/**
 * @param {{ title: string; unit?: string; ticks: LegendTick[]; colorStops: ColorStop[]; note?: string; footnote?: string; citation: string; accentRgb?: [number, number, number] }} config
 * @param {{ lede?: string; fileLabel?: string; fileName?: string }} [opts]
 */
export function buildFloatingLegendInnerHtml(config, opts = {}) {
  const unitSuffix = config.unit ? ` (${config.unit})` : "";
  const accent = config.accentRgb ? `rgb(${config.accentRgb.join(",")})` : "rgba(120, 175, 220, 0.85)";
  const gradient = legendConfigGradientCss(config, "to right");
  const tickHtml = buildHorizontalTicksHtml(config);

  const lede = opts.lede
    ? `<p class="sci-legend__lede">${escapeLegendHtml(opts.lede)}</p>`
    : "";
  const note = config.note
    ? `<p class="sci-legend__note">${escapeLegendHtml(config.note)}</p>`
    : "";
  const footnote = config.footnote
    ? `<p class="sci-legend__note">${escapeLegendHtml(config.footnote)}</p>`
    : "";
  const fileLine =
    opts.fileName != null
      ? `<p class="sci-legend__file"><span>Data</span> <code>${escapeLegendHtml(opts.fileName)}</code></p>`
      : "";

  return `
    <div class="sci-legend" style="--sci-accent:${accent}" role="group">
      <h3 class="sci-legend__title">${escapeLegendHtml(config.title)}${escapeLegendHtml(unitSuffix)}</h3>
      ${lede}
      <div class="sci-legend__scale-h" role="img" aria-label="Color scale for ${escapeLegendHtml(config.title)}">
        <div class="sci-legend__bar-h" style="background:${gradient}"></div>
        <div class="sci-legend__ticks-h">${tickHtml}</div>
      </div>
      ${note}
      ${footnote}
      ${fileLine}
      <p class="sci-legend__cite"><em>${escapeLegendHtml(config.citation)}</em></p>
    </div>`;
}

/**
 * @param {"suitability" | "layer"} kind
 * @param {{ layerKey?: string; overlayFile?: string }} ctx
 */
export function renderMapLegendFloatHtml(kind, ctx = {}) {
  if (kind === "suitability") {
    const file = ctx.overlayFile ? String(ctx.overlayFile).split("?")[0].split("#")[0] : "mars_landing_suitability_ml.tif";
    return buildFloatingLegendInnerHtml(SUITABILITY_LEGEND, {
      lede: "Higher % = better under VANGUARD engineering criteria.",
      fileName: file.split("/").pop() || file,
    });
  }
  const cfg = getScientificLegendConfig(ctx.layerKey);
  if (!cfg) return "";
  const info = ctx.datasetMeta;
  const lede = info?.description ? String(info.description) : "";
  const fileName = info?.file ? String(info.file).split("/").pop() : undefined;
  return buildFloatingLegendInnerHtml(cfg, { lede, fileName });
}
