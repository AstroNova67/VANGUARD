# Gap-fill score comparison (A / B / C)

Three arms, same `/predict` pipeline:

- **A · No corrections** — `data_gap_unfilled/`
- **B · Manual (IDW)** — `data_gap_filled/`
- **C · ML fill** — `data_gap_ml_filled/`

- Mean Δ(B−A): **-5.4796774193548385**
- Mean Δ(C−A): **-5.629677419354839**
- Mean Δ(C−B): **-0.15**
- Famous mean Δ(B−A) / Δ(C−A): **-0.5546666666666668** / **-0.5800000000000001**
- Gap-site mean Δ(B−A) / Δ(C−A): **-10.096874999999999** / **-10.36375**

## Largest |Δ(C−A)|

| Site | A | B (IDW) | C (ML) | Δ IDW | Δ ML | Δ ML−IDW |
|------|---|---------|--------|-------|-------|----------|
| Former gap (-50.86°N, 151.53°E) | 64.31 | 29.49 | 29.7 | -34.82 | -34.61 | +0.21 |
| Former gap (-79.11°N, -2.90°E) | 37.47 | 17.6 | 17.58 | -19.87 | -19.89 | -0.02 |
| Former gap (33.63°N, 75.28°E) | 58.48 | 35.43 | 38.98 | -23.05 | -19.50 | +3.55 |
| Former gap (-82.94°N, -127.08°E) | 34.6 | 17.96 | 17.75 | -16.64 | -16.85 | -0.21 |
| Former gap (-81.74°N, -63.06°E) | 32.94 | 16.59 | 16.57 | -16.35 | -16.37 | -0.02 |
| Former gap (-88.44°N, 173.08°E) | 31.47 | 16.35 | 16.53 | -15.12 | -14.94 | +0.18 |
| Former gap (-13.76°N, -126.11°E) | 58.64 | 44.48 | 44.48 | -14.16 | -14.16 | +0.00 |
| Former gap (-85.09°N, 126.76°E) | 44.11 | 30.13 | 30.19 | -13.98 | -13.92 | +0.06 |
| Viking 1 (Chryse Planitia) | 57.33 | 45.78 | 45.79 | -11.55 | -11.54 | +0.01 |
| Former gap (-55.89°N, 171.80°E) | 45.66 | 35.34 | 35.26 | -10.32 | -10.40 | -0.08 |

Full JSON: `scripts/gap_fill_score_compare.json`.

```bash
.venv/bin/python scripts/compare_gap_fill_scores.py
```
