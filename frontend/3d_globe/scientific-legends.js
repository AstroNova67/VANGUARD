/**
 * Scientific display ranges, colormaps, and floating legend markup for Mars GeoTIFF overlays.
 * Ranges tuned to `public/data_gap_filled/` (Mars Geoscience exports + IDW fills) so globe tint
 * uses the product’s actual dynamic range instead of outdated full-mission extremes.
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
    { value: 0, num: "0%", qual: "Very poor" },
    { value: 20, num: "20%", qual: "Poor" },
    { value: 35, num: "35%", qual: "Fair" },
    { value: 50, num: "50%", qual: "Good" },
    { value: 60, num: "60%", qual: "Excellent" },
  ],
  note: "Bands match live scores: Good ≥50, Excellent ≥60 (Meridiani ~64% is top-tier under this rubric).",
  citation: "Golombek et al. 2012; NASA Mars 2020 landing constraints",
  accentRgb: [255, 180, 80],
};

/** Globe surface layer keys → scientific legend config. */
export const SCIENTIFIC_LAYER_LEGENDS = {
  slope: {
    title: "Slope",
    unit: "°",
    // Gap-filled MOLA/HRSC blend (low-res): ~0–20°. Cap at 15° so contrast is visible.
    displayMin: 0,
    displayMax: 15,
    colorStops: [
      { t: 0, rgb: [46, 125, 50] },
      { t: 0.1, rgb: [102, 187, 106] },
      { t: 0.25, rgb: [174, 213, 129] },
      { t: 0.5, rgb: [255, 183, 77] },
      { t: 1, rgb: [198, 40, 40] },
    ],
    ticks: [
      { value: 0, num: "0°", qual: "Flat" },
      { value: 2, num: "2°", qual: "Very gentle" },
      { value: 5, num: "5°", qual: "Rover-safe" },
      { value: 10, num: "10°", qual: "Moderate" },
      { value: 15, num: "15°+", qual: "Steep (this map)" },
    ],
    footnote: "This export tops out near ~20°. Higher-res products can exceed NASA’s 30° EDL cutoff.",
    citation: "Golombek et al. 2012; Anderson et al. 2003 (MER)",
    accentRgb: [102, 187, 106],
  },
  thermalInertiaObs: {
    title: "Thermal inertia",
    unit: "TIU",
    // Dayside TES: bulk of values <450 TIU; keep headroom to 500.
    displayMin: 0,
    displayMax: 500,
    colorStops: [
      { t: 0, rgb: [30, 80, 165] },
      { t: 0.15, rgb: [70, 130, 210] },
      { t: 0.4, rgb: [230, 140, 55] },
      { t: 0.7, rgb: [240, 200, 150] },
      { t: 1, rgb: [252, 250, 248] },
    ],
    ticks: [
      { value: 0, num: "<50", qual: "Fine dust" },
      { value: 100, num: "100", qual: "MSL minimum" },
      { value: 200, num: "200", qual: "Sandy mix" },
      { value: 350, num: "350", qual: "Rock mix" },
      { value: 500, num: "500+", qual: "Bedrock / high" },
    ],
    footnote: "Jezero typical ~200–300 TIU (Ahern et al. 2021). Rare pixels exceed 500.",
    citation: "Putzig & Mellon 2007; Golombek et al. 2012 (MSL)",
    accentRgb: [230, 140, 55],
  },
  temperature: {
    title: "Surface temperature",
    unit: "°C",
    // Yearly avg °C product in data_gap_filled: about −125 to −30.
    displayMin: -130,
    displayMax: -25,
    colorStops: [
      { t: 0, rgb: [12, 28, 110] },
      { t: 0.28, rgb: [40, 90, 175] },
      { t: 0.55, rgb: [120, 150, 200] },
      { t: 0.78, rgb: [210, 140, 90] },
      { t: 1, rgb: [255, 130, 50] },
    ],
    ticks: [
      { value: -130, num: "−130°C", qual: "Coldest (this map)" },
      { value: -100, num: "−100°C", qual: "Polar winter" },
      { value: -80, num: "−80°C", qual: "High latitude" },
      { value: -50, num: "−50°C", qual: "Mid-latitude" },
      { value: -25, num: "−25°C", qual: "Warmest (this map)" },
    ],
    citation: "NASA Mars 2020 constraints; Viking / Curiosity / Perseverance",
    accentRgb: [210, 140, 90],
  },
  tempRange: {
    title: "Temperature range",
    unit: "°C",
    displayMin: 70,
    displayMax: 170,
    colorStops: [
      { t: 0, rgb: [40, 90, 160] },
      { t: 0.35, rgb: [90, 160, 200] },
      { t: 0.65, rgb: [230, 170, 70] },
      { t: 1, rgb: [200, 60, 40] },
    ],
    ticks: [
      { value: 70, num: "70°C", qual: "Low swing" },
      { value: 100, num: "100°C", qual: "Moderate" },
      { value: 130, num: "130°C", qual: "Typical" },
      { value: 155, num: "155°C", qual: "High swing" },
      { value: 170, num: "170°C", qual: "Extreme" },
    ],
    citation: "Yearly surface temperature variation product (Mars Geoscience)",
    accentRgb: [230, 170, 70],
  },
  ferric: {
    title: "OMEGA ferric / dust",
    unit: "index",
    // Filled OMEGA ferric is tightly clustered (~0.90–1.04); old 0–2 scale washed out contrast.
    displayMin: 0.9,
    displayMax: 1.04,
    colorStops: [
      { t: 0, rgb: [56, 142, 72] },
      { t: 0.25, rgb: [120, 185, 100] },
      { t: 0.5, rgb: [210, 190, 90] },
      { t: 0.75, rgb: [220, 110, 70] },
      { t: 1, rgb: [175, 35, 35] },
    ],
    ticks: [
      { value: 0.9, num: "0.90", qual: "Lower dust" },
      { value: 0.95, num: "0.95", qual: "Rocky mix" },
      { value: 0.99, num: "0.99", qual: "Typical" },
      { value: 1.02, num: "1.02", qual: "Dustier" },
      { value: 1.04, num: "1.04", qual: "Highest (this map)" },
    ],
    footnote: "Scale matches the gap-filled OMEGA ferric export, not a 0–2 lab index.",
    citation: "Ody et al. 2012 (OMEGA); Golombek et al. 2012",
    accentRgb: [120, 185, 100],
  },
  elevation: {
    title: "Elevation (MOLA)",
    unit: "m",
    displayMin: -8200,
    displayMax: 21000,
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
      { value: 4000, num: "+4 km", qual: "Highlands" },
      { value: 21000, num: "+21 km", qual: "Olympus" },
    ],
    footnote: "Mars 2020 target elevation < −0.5 km (northern lowlands).",
    citation: "Smith et al. 1999 (MOLA); NASA Mars 2020",
    accentRgb: [90, 165, 110],
  },
  grsWaterWt: {
    title: "GRS water equivalent",
    unit: "% wt",
    // Odyssey GRS MONS: mid-lats ~2–15%; polar cells approach 100% in this export.
    displayMin: 0,
    displayMax: 40,
    colorStops: [
      { t: 0, rgb: [205, 178, 145] },
      { t: 0.2, rgb: [165, 175, 195] },
      { t: 0.45, rgb: [90, 140, 200] },
      { t: 0.75, rgb: [40, 90, 180] },
      { t: 1, rgb: [20, 50, 130] },
    ],
    ticks: [
      { value: 0, num: "0%", qual: "Dry" },
      { value: 5, num: "5%", qual: "Typical mid-lat" },
      { value: 10, num: "10%", qual: "Elevated" },
      { value: 20, num: "20%", qual: "High" },
      { value: 40, num: "40%+", qual: "Polar / ice-rich" },
    ],
    note: "GRS footprint ~520 km — regional averages. Values above ~40% are clipped in color only.",
    citation: "Feldman et al. 2004; Boynton et al. 2002 (Odyssey GRS)",
    accentRgb: [90, 140, 200],
  },
  roughness: {
    title: "Roughness (0.6 km)",
    unit: "m RMS",
    // Numeric roughness export: ~64–255 (not 0–500 metres).
    displayMin: 60,
    displayMax: 220,
    colorStops: [
      { t: 0, rgb: [56, 142, 72] },
      { t: 0.25, rgb: [120, 190, 110] },
      { t: 0.55, rgb: [230, 190, 80] },
      { t: 1, rgb: [192, 48, 42] },
    ],
    ticks: [
      { value: 60, num: "60", qual: "Smoothest" },
      { value: 100, num: "100", qual: "Smooth" },
      { value: 140, num: "140", qual: "Typical" },
      { value: 180, num: "180", qual: "Rougher" },
      { value: 220, num: "220+", qual: "Roughest" },
    ],
    footnote: "Scale matches the gap-filled MOLA 0.6 km numeric product.",
    citation: "Neumann et al. 2003 (MOLA); JPL roughness map",
    accentRgb: [120, 190, 110],
  },
  albedo: {
    title: "OMEGA albedo",
    unit: "",
    displayMin: 0.1,
    displayMax: 0.55,
    colorStops: [
      { t: 0, rgb: [30, 30, 35] },
      { t: 0.25, rgb: [90, 70, 55] },
      { t: 0.5, rgb: [170, 130, 90] },
      { t: 0.75, rgb: [220, 195, 150] },
      { t: 1, rgb: [245, 240, 230] },
    ],
    ticks: [
      { value: 0.1, num: "0.10", qual: "Dark" },
      { value: 0.2, num: "0.20", qual: "Low" },
      { value: 0.3, num: "0.30", qual: "Typical" },
      { value: 0.4, num: "0.40", qual: "Bright" },
      { value: 0.55, num: "0.55+", qual: "Very bright" },
    ],
    citation: "OMEGA R1080 albedo (Mars Geoscience export)",
    accentRgb: [170, 130, 90],
  },
  lambertAlbedo: {
    title: "TES Lambert albedo",
    unit: "",
    displayMin: 0.08,
    displayMax: 0.3,
    colorStops: [
      { t: 0, rgb: [30, 30, 35] },
      { t: 0.3, rgb: [100, 80, 60] },
      { t: 0.6, rgb: [180, 150, 110] },
      { t: 1, rgb: [240, 230, 210] },
    ],
    ticks: [
      { value: 0.08, num: "0.08", qual: "Dark" },
      { value: 0.15, num: "0.15", qual: "Low" },
      { value: 0.2, num: "0.20", qual: "Typical" },
      { value: 0.26, num: "0.26", qual: "Bright" },
      { value: 0.3, num: "0.30", qual: "Brightest" },
    ],
    citation: "TES Lambert albedo numeric product",
    accentRgb: [180, 150, 110],
  },
  crustalThickness: {
    title: "Crustal thickness",
    unit: "km",
    displayMin: 15,
    displayMax: 90,
    colorStops: [
      { t: 0, rgb: [40, 80, 160] },
      { t: 0.35, rgb: [80, 160, 180] },
      { t: 0.65, rgb: [210, 180, 90] },
      { t: 1, rgb: [180, 70, 50] },
    ],
    ticks: [
      { value: 15, num: "15 km", qual: "Thin" },
      { value: 30, num: "30 km", qual: "Low" },
      { value: 45, num: "45 km", qual: "Typical" },
      { value: 60, num: "60 km", qual: "Thick" },
      { value: 90, num: "90 km", qual: "Thickest" },
    ],
    citation: "GMM-3 crustal thickness (Wieczorek / Mars Geoscience)",
    accentRgb: [80, 160, 180],
  },
  pyroxene: {
    title: "Pyroxene (OMEGA)",
    unit: "index",
    displayMin: 0,
    displayMax: 0.06,
    colorStops: [
      { t: 0, rgb: [40, 50, 70] },
      { t: 0.35, rgb: [80, 120, 90] },
      { t: 0.7, rgb: [180, 140, 70] },
      { t: 1, rgb: [200, 80, 40] },
    ],
    ticks: [
      { value: 0, num: "0", qual: "None / nodata" },
      { value: 0.02, num: "0.02", qual: "Low" },
      { value: 0.04, num: "0.04", qual: "Moderate" },
      { value: 0.06, num: "0.06+", qual: "Higher" },
    ],
    footnote: "Zeros may be real absence or no coverage — not gap-filled.",
    citation: "OMEGA BD2000 pyroxene index",
    accentRgb: [180, 140, 70],
  },
  basalt: {
    title: "TES basalt",
    unit: "%",
    displayMin: 0,
    displayMax: 60,
    colorStops: [
      { t: 0, rgb: [45, 45, 50] },
      { t: 0.25, rgb: [90, 70, 60] },
      { t: 0.55, rgb: [160, 100, 70] },
      { t: 1, rgb: [210, 90, 50] },
    ],
    ticks: [
      { value: 0, num: "0%", qual: "None / nodata" },
      { value: 15, num: "15%", qual: "Low" },
      { value: 30, num: "30%", qual: "Moderate" },
      { value: 45, num: "45%", qual: "High" },
      { value: 60, num: "60%+", qual: "Highest" },
    ],
    footnote: "Zeros may be real absence or no coverage — not gap-filled.",
    citation: "TES Basalt numeric abundance",
    accentRgb: [160, 100, 70],
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
