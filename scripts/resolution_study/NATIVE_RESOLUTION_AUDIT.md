# Native resolution audit — `frontend/3d_globe/public/data/`

Audit date: 2026-05-20. Pixel sizes from rasterio; global equirectangular expectation: **height ≈ width ÷ 2**.

## Task 1 — Native resolution table

| Filename | Width × Height | Effective PPD (W÷360) | Georeferencing | JMARS / mission product | Global grid notes |
|----------|----------------|----------------------:|----------------|-------------------------|-------------------|
| `MOLA_128ppd_topo.tif` | 2880 × 773 | 8.00 | Identity / none | MOLA 128 ppd topography (elevation) | **Likely partial export** — expected H≈1440, got 773 (~46% of 2:1). **Georef not attached.** |
| `mola_hrsc_blend_slope_v2.tif` | 1214 × 562 | 3.37 | Identity / none | MOLA/HRSC blended slope v2 | Mild aspect mismatch (H≈607 for 2:1); treated as near-global JMARS export. Georef copy in `scripts/georef_layers/`. |
| `mola_roughness_0.6km_numeric.tif` | 1214 × 562 | 3.37 | Identity / none | MOLA roughness 0.6 km | Same as slope group. |
| `omega_albedo_r1080.tif` | 1214 × 562 | 3.37 | Identity / none | OMEGA albedo R1080 | Same as slope group. |
| `mars_yearly_avg_temperature_celsius.tif` | 1214 × 562 | 3.37 | Identity / none | Yearly average surface temperature (°C) | Same — **coarsest scored surface temp**. |
| `mars_yearly_temperature_range_v1.0.tif` | 1214 × 562 | 3.37 | Identity / none | Yearly temperature range | Same as slope group. |
| `mars_crustal_thickness_gmm3_rm1.tif` | 1214 × 562 | 3.37 | Identity / none | Crustal thickness (g/cm³, RM1) | Same as slope group. |
| `omega_ferric_nnphs.tif` | 1214 × 562 | 3.37 | Identity / none | OMEGA ferric/dust NNPHS | Same — **coarsest scored dust/ferric**. |
| `omega_pyroxene_bd2000.tif` | 1214 × 562 | 3.37 | Identity / none | OMEGA pyroxene BD2000 | Same as slope group. |
| `TES_Basalt_numeric.tif` | 1214 × 562 | 3.37 | Identity / none | TES basalt abundance | Same as slope group. |
| `TES_Lambert_Albedo_numeric.tif` | 1214 × 562 | 3.37 | Identity / none | TES Lambert albedo | Same as slope group. |
| `tes_dayside_ti_putzig_2007.tif` | 3068 × 1548 | 8.52 | Identity / none | TES dayside thermal inertia (Putzig & Mellon 2007) | ~2:1 (H exp. 1534); georef copy in `scripts/georef_layers/`. |
| `mars_odyssey_grs_mons_perc_wt.tif` | 3068 × 1548 | 8.52 | Identity / none | Odyssey GRS MONS water equivalent (% wt) | Same as TI. |
| `mars_global_input_stack_32ppd.tif` | 11520 × 5760 | 32.00 | **Yes** — Mars lon/lat, 0.03125°/px | 13-band aligned stack for batch ML | Correct global 2:1 @ 32 PPD. |
| `mars_landing_suitability_ml.tif` | 11520 × 5760 | 32.00 | **Yes** — same grid as stack | ML landing suitability 0–100% | Correct global 2:1 @ 32 PPD. |

## Task 4 — Native resolution recommendation

**Coarsest native PPD among the five scored properties** (slope, dust/ferric, surface temperature, thermal inertia, water/GRS):

| Property | Source file | Native PPD (lon) |
|----------|-------------|-----------------:|
| Slope | `mola_hrsc_blend_slope_v2.tif` | **~3.4** |
| Dust / ferric | `omega_ferric_nnphs.tif` | **~3.4** |
| Surface temperature | `mars_yearly_avg_temperature_celsius.tif` | **~3.4** |
| Thermal inertia | `tes_dayside_ti_putzig_2007.tif` | ~8.5 |
| Water / GRS | `mars_odyssey_grs_mons_perc_wt.tif` | ~8.5 |

The scientifically honest effective resolution of a global suitability map built from these natives is therefore about **3–4 PPD** (~0.29–0.33°/pixel), set by the 1214×562 JMARS bundle that constrains slope, ferric/dust, and yearly-average temperature in the stack after alignment. Publishing or interpreting a **32 PPD** suitability GeoTIFF does not add independent information for those three inputs—it upsamples them by roughly **8–10×** along with auxiliary bands. A **4 PPD** (~1440×720) or **8 PPD** (~2880×1440) global product is a better balance: it stays near or slightly above the coarsest constraint (4 PPD is ~1.2× the native spacing of the 1214×562 layers; 8 PPD captures TI/GRS native spacing without the storage and inference cost of 32 PPD). Use **32 PPD** only when pixel-level agreement with the existing stack, globe overlay, and batch pipeline is required; for science communication and regional ranking, **4–8 PPD** is the defensible maximum.

## Task 5 — Stack and suitability georeferencing

Both files already have correct metadata:

- **CRS:** Mars geographic (`+proj=longlat +R=3396190 +no_defs`)
- **Transform:** `| 0.03125, 0, -180 | / | 0, -0.03125, 90 |` (32 PPD, north-up, −180…180°, −90…90°)
- **Bounds:** (−180, −90, 180, 90)

No changes were written to `public/data/` for these two files.

## Task 2 — Georef copies

Run: `uv run python scripts/attach_mars_georef_layers.py`

Outputs: `scripts/georef_layers/` (12 files; `MOLA_128ppd_topo.tif` skipped pending extent confirmation).

## Task 3 — Resolution study

Run: `uv run python scripts/generate_resolution_study.py`

Outputs: `scripts/resolution_study/stacks/`, `scripts/resolution_study/outputs/`, `scripts/resolution_study/comparison_report.md`
