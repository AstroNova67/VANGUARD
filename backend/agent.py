"""
VANGUARD Mars assistant (OpenAI Agents SDK).

Tools: famous-site coordinates, README documentation, landing analysis at lat/lon.

CLI:  uv run python backend/agent.py
API:  POST /agent/chat  (when Flask app is running)
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from agents.items import ItemHelpers, MessageOutputItem, ToolCallItem, ToolCallOutputItem
from agents.run_internal.error_handlers import format_final_output_text
from openai import AsyncOpenAI

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from backend.agent_schema import AgentReply
    from backend.mars_landing_search import find_best_landing_site
    from backend.mars_raster import sample_mars_data_at
    from backend.mars_sites import MARS_FAMOUS_SITES, MarsSite, find_mars_site, format_sites_list
    from backend.landing_predict import compute_landing_prediction, score_band_label
except ImportError:
    from agent_schema import AgentReply
    from mars_landing_search import find_best_landing_site
    from mars_raster import sample_mars_data_at
    from mars_sites import MARS_FAMOUS_SITES, MarsSite, find_mars_site, format_sites_list
    from landing_predict import compute_landing_prediction, score_band_label

load_dotenv(PROJECT_ROOT / ".env", override=True)

logger = logging.getLogger("vanguard.agent")


def _agent_trace_enabled() -> bool:
    return os.environ.get("VANGUARD_AGENT_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _trace(msg: str, **fields: Any) -> None:
    """Structured trace lines to stderr (enable with VANGUARD_AGENT_TRACE=1)."""
    if not _agent_trace_enabled():
        return
    suffix = ""
    if fields:
        suffix = " " + json.dumps(fields, default=str, ensure_ascii=False)
    print(f"[vanguard.agent] {msg}{suffix}", file=sys.stderr, flush=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is not set (add it to .env at the repo root).", file=sys.stderr)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
openai_model = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client)

INSTRUCTIONS = """
You are the VANGUARD Mars landing-site assistant.

VANGUARD samples Mars GeoTIFFs at lat/lon, runs ML models, and computes a landing suitability % using
weighted slope, dust, surface temperature, thermal inertia, and water. Custom scoring_weights come from the UI.

Tools (always use tools for scores — never invent numbers):
- find_best_landing_region: when the user wants a good landing **region** or **site** without naming a mission.
  Returns ONE globally best candidate (ML map search + re-score with active weights). Do NOT answer with a list of
  famous craters unless they explicitly ask to compare named missions.
- focus_mars_site / focus_mars_coordinates: user names a place or gives coordinates to show on the globe.
- analyze_landing_site: numbers only at lat/lon.
- compare_landing_sites: only when the user explicitly asks to compare named sites (Jezero, Gale, etc.).
- get_scoring_weights / set_scoring_weights: weight changes; re-run prediction after set_scoring_weights.

Output format (required — you have structured output_type AgentReply; fill every field the schema expects):
- summary: 1–3 short plain sentences only. No HTML tags, no <br>, no numbered lists, no markdown headings.
- intent: best_region when recommending find_best_landing_region result; analyze_site / navigate / compare / explain / settings / general otherwise.
- best_site: fill from find_best_landing_region or focus/analyze tool when recommending one location; null otherwise.
- scoring_weights_percent: copy from tool output when discussing scoring.
- top_contributions: use contribution_percent from score_breakdown (not weight percents).

Latitude °N (−90…90). Longitude °E (−180…180).
"""

# Collected per /agent/chat request for the browser to execute (focus globe, change layer, etc.).
_ui_actions: contextvars.ContextVar[list[dict[str, Any]]] = contextvars.ContextVar(
    "vanguard_ui_actions", default=[]
)

_scoring_weights: contextvars.ContextVar[dict[str, float] | None] = contextvars.ContextVar(
    "vanguard_scoring_weights", default=None
)


def _queue_ui_action(action: dict[str, Any]) -> None:
    try:
        _ui_actions.get().append(action)
        _trace("ui_action queued", action=action)
    except LookupError:
        pass


def _parse_agent_reply(result: Any, agent: Agent) -> AgentReply:
    """
    Read structured output from ``Runner.run`` → ``RunResult.final_output``.

    With ``output_type=AgentReply``, the SDK validates model JSON into
    ``AgentReply`` before returning (Agents SDK structured outputs).
    Fallbacks cover handoffs or rare plain-text responses.
    """
    fo = result.final_output
    if isinstance(fo, AgentReply):
        return fo
    if isinstance(fo, dict):
        return AgentReply.model_validate(fo)
    if fo is None:
        return AgentReply(summary="No response produced.", intent="general")
    if isinstance(fo, str):
        stripped = fo.strip()
        if stripped.startswith("{"):
            try:
                return AgentReply.model_validate(json.loads(stripped))
            except Exception:
                pass
        return AgentReply(summary=stripped or "Done.", intent="general")
    return AgentReply(summary=str(fo).strip() or "Done.", intent="general")


def _extract_reply_text(result: Any, agent: Agent) -> str:
    """
    Normalize Agents SDK RunResult.final_output to a user-facing string.
    Avoids returning raw dicts to the browser (which show as '[object Object]' in JS).
    """
    fo = result.final_output
    if isinstance(fo, str):
        text = fo.strip()
        if text:
            return fo

    try:
        formatted = format_final_output_text(agent, fo)
        if isinstance(formatted, str) and formatted.strip():
            return formatted
    except Exception as exc:
        _trace("format_final_output_text failed", error=str(exc))

    message_text = ItemHelpers.text_message_outputs(result.new_items).strip()
    if message_text:
        return message_text

    if fo is None:
        return ""
    if isinstance(fo, dict):
        for key in ("text", "content", "message", "reply", "response"):
            val = fo.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return json.dumps(fo, indent=2, ensure_ascii=False)
    if isinstance(fo, list):
        return json.dumps(fo, indent=2, ensure_ascii=False)
    return str(fo)


def _trace_run_result(user_message: str, result: Any, reply: str, actions: list[dict[str, Any]]) -> None:
    if not _agent_trace_enabled():
        return
    _trace(
        "turn complete",
        user_preview=user_message[:160],
        item_count=len(result.new_items),
        final_output_type=type(result.final_output).__name__,
        reply_chars=len(reply),
        ui_action_count=len(actions),
    )
    for idx, item in enumerate(result.new_items):
        kind = type(item).__name__
        if isinstance(item, ToolCallItem):
            _trace(
                f"  item[{idx}] tool_call",
                tool=item.tool_name,
                call_id=item.call_id,
            )
        elif isinstance(item, ToolCallOutputItem):
            preview = str(item.output)
            if len(preview) > 240:
                preview = preview[:240] + "…"
            _trace(f"  item[{idx}] tool_output", preview=preview)
        elif isinstance(item, MessageOutputItem):
            preview = ItemHelpers.text_message_output(item)
            if len(preview) > 200:
                preview = preview[:200] + "…"
            _trace(f"  item[{idx}] assistant_message", preview=preview)
        else:
            _trace(f"  item[{idx}] {kind}")
    if actions:
        _trace("ui_actions", actions=actions)
    if reply:
        _trace("reply_preview", text=reply[:280] + ("…" if len(reply) > 280 else ""))


# Globe layer keys match frontend/3d_globe/index.js `marsDatasets` (+ "photo").
GLOBE_LAYER_KEYS: frozenset[str] = frozenset(
    {
        "photo",
        "elevation",
        "slope",
        "roughness",
        "albedo",
        "temperature",
        "tempRange",
        "crustalThickness",
        "ferric",
        "pyroxene",
        "basalt",
        "lambertAlbedo",
        "thermalInertiaObs",
        "grsWaterWt",
    }
)


def _readme_excerpt(max_chars: int = 14_000) -> str:
    path = PROJECT_ROOT / "README.md"
    if not path.is_file():
        return "README.md not found in the project root."
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n… [truncated; full README is {len(text)} characters]"


def _active_scoring_weights() -> dict[str, float] | None:
    """Weights from this chat request or from set_scoring_weights during the turn."""
    return _scoring_weights.get()


def _run_prediction(mars_data: dict) -> dict:
    """Prefer in-process predict when Flask models are loaded; else call the HTTP API."""
    try:
        from backend.scoring import parse_scoring_weights
    except ImportError:
        from scoring import parse_scoring_weights
    weights = parse_scoring_weights(_active_scoring_weights())
    try:
        from backend import app as vanguard_app
    except ImportError:
        import app as vanguard_app  # type: ignore

    if getattr(vanguard_app, "models_loaded", False):
        return compute_landing_prediction(
            mars_data,
            models_loaded=True,
            regression_models=vanguard_app.regression_models,
            scoring_weights=weights,
        )

    api_base = os.getenv("VANGUARD_API_BASE", "http://127.0.0.1:5002").rstrip("/")
    try:
        import requests

        payload = dict(mars_data)
        payload["scoring_weights"] = weights
        resp = requests.post(f"{api_base}/predict", json=payload, timeout=180)
        data = resp.json()
        if resp.status_code == 503:
            return {
                "success": False,
                "error": data.get("error", "Models still loading on the API server."),
            }
        if not resp.ok:
            return {"success": False, "error": data.get("error", resp.text)}
        if data.get("success") and "score_interpretation" not in data:
            data["score_interpretation"] = score_band_label(float(data.get("landing_score", 0)))
        return data
    except Exception as e:
        return {
            "success": False,
            "error": (
                f"Could not reach VANGUARD API at {api_base}/predict ({e}). "
                "Start the server with ./start_api.sh or uv run python backend/app.py."
            ),
        }


def _summarize_prediction(result: dict) -> str:
    if not result.get("success"):
        return json.dumps(result, indent=2)

    score = result.get("landing_score")
    interp = result.get("score_interpretation") or score_band_label(float(score or 0))
    fused = (result.get("predictions") or {}).get("neural_networks") or {}
    baseline = (result.get("predictions") or {}).get("neural_networks_baseline") or {}
    reg = (result.get("predictions") or {}).get("regression_models") or {}
    raw = result.get("raw_mars_data") or {}
    sources = result.get("data_sources") or {}
    overrides = result.get("overrides_applied") or {}

    summary = {
        "landing_score_percent": score,
        "interpretation": interp,
        "coordinates": {"lat": raw.get("lat"), "lon": raw.get("lon")},
        "values_used_for_score": fused,
        "pure_neural_network_outputs": baseline,
        "xgboost_outputs": reg,
        "data_sources": sources,
        "model_overrides": overrides,
        "sampled_raster_fields": {
            k: raw[k]
            for k in (
                "elevation",
                "slope",
                "temperature",
                "thermalInertia",
                "ferric",
                "grsWaterWt",
                "albedo",
                "roughness",
            )
            if k in raw
        },
        "scoring_weights_fraction": result.get("scoring_weights"),
        "scoring_weights_percent": result.get("scoring_weights_percent"),
        "score_breakdown": result.get("score_breakdown"),
    }
    return json.dumps(summary, indent=2)


@function_tool
def get_scoring_weights() -> str:
    """
    Return the landing suitability scoring weights currently in effect (research defaults or user overrides).
    Weights sum to 1.0; percents are shown for readability.
    """
    try:
        from backend.scoring import (
            DEFAULT_SCORING_WEIGHTS,
            parse_scoring_weights,
            scoring_weights_as_percent,
        )
    except ImportError:
        from scoring import (
            DEFAULT_SCORING_WEIGHTS,
            parse_scoring_weights,
            scoring_weights_as_percent,
        )
    active = _active_scoring_weights()
    weights = parse_scoring_weights(active)
    return json.dumps(
        {
            "weights_fraction": weights,
            "weights_percent": scoring_weights_as_percent(weights),
            "is_custom": active is not None,
            "research_defaults_fraction": DEFAULT_SCORING_WEIGHTS,
            "research_defaults_percent": scoring_weights_as_percent(DEFAULT_SCORING_WEIGHTS),
            "keys": ["slope", "dust", "surface_temp", "thermal_inertia", "water"],
        },
        indent=2,
    )


@function_tool
def set_scoring_weights(
    slope: float,
    dust: float,
    surface_temp: float,
    thermal_inertia: float,
    water: float,
) -> str:
    """
    Set custom landing suitability weights for this conversation (must sum to 100 if using percents, or 1.0 if fractions).
    slope: terrain slope importance (default 30%).
    dust: ferric/dust index importance (default 20%).
    surface_temp: surface temperature importance (default 20%).
    thermal_inertia: thermal inertia importance (default 20%).
    water: GRS water equivalent importance (default 10%).
    """
    try:
        from backend.scoring import parse_scoring_weights, scoring_weights_as_percent
    except ImportError:
        from scoring import parse_scoring_weights, scoring_weights_as_percent

    raw = {
        "slope": slope,
        "dust": dust,
        "surface_temp": surface_temp,
        "thermal_inertia": thermal_inertia,
        "water": water,
    }
    try:
        normalized = parse_scoring_weights(raw)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    _scoring_weights.set(normalized)
    _trace("scoring_weights set", weights=scoring_weights_as_percent(normalized))
    return json.dumps(
        {
            "success": True,
            "message": "Scoring weights updated for this session. Re-analyze a site to apply them.",
            "weights_fraction": normalized,
            "weights_percent": scoring_weights_as_percent(normalized),
        },
        indent=2,
    )


@function_tool
def list_mars_sites() -> str:
    """List famous Mars locations with latitude and longitude (missions, craters, volcanoes)."""
    return format_sites_list()


@function_tool
def lookup_mars_site(site_query: str) -> str:
    """
    Look up latitude and longitude for a named Mars site (e.g. 'Jezero', 'Gale', 'Hellas').
    site_query: site id or part of the display name.
    """
    site = find_mars_site(site_query)
    if site is None:
        ids = ", ".join(s.id for s in MARS_FAMOUS_SITES)
        return f"No site matched '{site_query}'. Try one of: {ids}"
    return (
        f"{site.name}\n"
        f"  id: {site.id}\n"
        f"  latitude: {site.lat}°N\n"
        f"  longitude: {site.lon}°E"
    )


@function_tool
def get_vanguard_documentation(topic: str = "") -> str:
    """
    Return VANGUARD project documentation from README.md (overview, API, scoring, datasets).
    topic: optional keyword to focus on (e.g. 'scoring', 'predict', 'globe'); leave empty for full excerpt.
    """
    text = _readme_excerpt()
    if not topic.strip():
        return text
    needle = topic.strip().lower()
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if needle in line.lower()]
    if not hits:
        return (
            f"No section header line matched '{topic}'. Full README excerpt:\n\n{text[:8000]}"
        )
    chunks = []
    for i in hits[:5]:
        start = max(0, i - 2)
        end = min(len(lines), i + 40)
        chunks.append("\n".join(lines[start:end]))
    return "\n\n---\n\n".join(chunks)


@function_tool
def focus_mars_site(
    site_query: str,
    run_prediction: bool = True,
    tight_camera: bool = False,
) -> str:
    """
    Move the user's globe to a famous Mars site, load all raster layers at that point, and optionally run landing prediction.
    site_query: site id or name (e.g. 'gale', 'Jezero', 'Hellas').
    run_prediction: if true, trigger the same landing suitability run as the Predict button.
    tight_camera: if true, zoom slightly closer (small craters / landers).
    """
    site = find_mars_site(site_query)
    if site is None:
        return json.dumps(
            {"success": False, "error": f"No site matched '{site_query}'."}
        )
    _queue_ui_action(
        {
            "type": "focus",
            "lat": site.lat,
            "lon": site.lon,
            "siteId": site.id,
            "siteName": site.name,
            "runPrediction": bool(run_prediction),
            "tightCamera": bool(tight_camera),
        }
    )
    _trace("focus_mars_site", site=site.id, lat=site.lat, lon=site.lon)
    mars_data = sample_mars_data_at(site.lat, site.lon)
    result = _run_prediction(mars_data)
    payload = json.loads(_summarize_prediction(result))
    payload["ui_focused"] = site.id
    return json.dumps(payload, indent=2)


@function_tool
def focus_mars_coordinates(
    latitude: float,
    longitude: float,
    run_prediction: bool = True,
    tight_camera: bool = False,
) -> str:
    """
    Move the globe to explicit coordinates, load rasters, and optionally run landing prediction.
    """
    lat = float(latitude)
    lon = float(longitude)
    if not (-90.0 <= lat <= 90.0):
        return json.dumps({"success": False, "error": "latitude must be between -90 and 90"})
    if not (-180.0 <= lon <= 180.0):
        return json.dumps({"success": False, "error": "longitude must be between -180 and 180"})
    _queue_ui_action(
        {
            "type": "focus",
            "lat": lat,
            "lon": lon,
            "runPrediction": bool(run_prediction),
            "tightCamera": bool(tight_camera),
        }
    )
    mars_data = sample_mars_data_at(lat, lon)
    result = _run_prediction(mars_data)
    return _summarize_prediction(result)


@function_tool
def set_globe_surface_layer(layer_key: str) -> str:
    """
    Change the globe surface visualization layer in the UI (same options as the Globe surface dropdown).
    layer_key: one of photo, elevation, slope, roughness, albedo, temperature, tempRange,
    crustalThickness, ferric, pyroxene, basalt, lambertAlbedo, thermalInertiaObs, grsWaterWt.
    """
    key = layer_key.strip()
    aliases = {
        "thermal_inertia": "thermalInertiaObs",
        "thermalinertia": "thermalInertiaObs",
        "ti": "thermalInertiaObs",
        "water": "grsWaterWt",
        "grs": "grsWaterWt",
        "dust": "ferric",
        "temp": "temperature",
        "mola": "elevation",
    }
    key = aliases.get(key.lower(), key)
    if key not in GLOBE_LAYER_KEYS:
        return json.dumps(
            {
                "success": False,
                "error": f"Unknown layer '{layer_key}'. Valid keys: {', '.join(sorted(GLOBE_LAYER_KEYS))}",
            }
        )
    _queue_ui_action({"type": "set_globe_layer", "layer": key})
    return json.dumps({"success": True, "layer": key})


@function_tool
def find_best_landing_region() -> str:
    """
    Search the global ML suitability map for strong candidates, re-score with current scoring weights,
    and return the single best latitude/longitude. Use when the user asks for a good landing region
  (not a tour of famous mission sites). Moves the globe to the result.
    """
    out = find_best_landing_site(_run_prediction)
    if not out.get("success"):
        return json.dumps(out)
    best = out["best"]
    _queue_ui_action(
        {
            "type": "focus",
            "lat": best["latitude"],
            "lon": best["longitude"],
            "runPrediction": True,
        }
    )
    _trace(
        "find_best_landing_region",
        lat=best["latitude"],
        lon=best["longitude"],
        score=best.get("landing_score_percent"),
    )
    return json.dumps(out, indent=2)


@function_tool
def compare_landing_sites(
    site_ids: str = "",
    max_sites: int = 8,
) -> str:
    """
    Compare landing suitability at multiple famous Mars sites using the current scoring weights.
    site_ids: optional comma-separated site ids (e.g. 'jezero,gale,meridiani'); empty uses common mission sites.
    max_sites: maximum number of sites to score (default 8).
    """
    cap = max(1, min(int(max_sites), len(MARS_FAMOUS_SITES)))
    sites: list[MarsSite] = []
    if site_ids.strip():
        for part in site_ids.split(","):
            q = part.strip()
            if not q:
                continue
            site = find_mars_site(q)
            if site is not None and site not in sites:
                sites.append(site)
    else:
        default_ids = (
            "jezero",
            "gale",
            "meridiani",
            "gusev",
            "viking1",
            "insight",
            "pathfinder",
            "hellas",
        )
        for sid in default_ids:
            site = find_mars_site(sid)
            if site is not None:
                sites.append(site)
    sites = sites[:cap]
    if not sites:
        return json.dumps(
            {
                "success": False,
                "error": "No sites matched. Use list_mars_sites or pass site_ids like 'jezero,gale'.",
            }
        )

    rows: list[dict[str, Any]] = []
    weights_percent: dict[str, float] | None = None
    for site in sites:
        mars_data = sample_mars_data_at(site.lat, site.lon)
        result = _run_prediction(mars_data)
        if not result.get("success"):
            rows.append(
                {
                    "site_id": site.id,
                    "name": site.name,
                    "error": result.get("error", "prediction failed"),
                }
            )
            continue
        if weights_percent is None:
            weights_percent = result.get("scoring_weights_percent")
        rows.append(
            {
                "site_id": site.id,
                "name": site.name,
                "lat": site.lat,
                "lon": site.lon,
                "landing_score_percent": result.get("landing_score"),
                "interpretation": result.get("score_interpretation"),
                "score_breakdown": result.get("score_breakdown"),
            }
        )

    ok_rows = [r for r in rows if "landing_score_percent" in r]
    ok_rows.sort(key=lambda r: float(r["landing_score_percent"] or 0), reverse=True)
    return json.dumps(
        {
            "success": True,
            "scoring_weights_percent": weights_percent,
            "sites_ranked_best_first": ok_rows,
            "sites_with_errors": [r for r in rows if "error" in r],
        },
        indent=2,
    )


@function_tool
def analyze_landing_site(latitude: float, longitude: float) -> str:
    """
    Sample Mars GeoTIFFs at the given coordinates and run VANGUARD landing suitability prediction.
    latitude: degrees north (−90 to 90).
    longitude: degrees east (−180 to 180).
    Returns JSON with landing score, property values, and model details.
    """
    lat = float(latitude)
    lon = float(longitude)
    if not (-90.0 <= lat <= 90.0):
        return json.dumps({"success": False, "error": "latitude must be between -90 and 90"})
    if not (-180.0 <= lon <= 180.0):
        return json.dumps({"success": False, "error": "longitude must be between -180 and 180"})

    mars_data = sample_mars_data_at(lat, lon)
    result = _run_prediction(mars_data)
    return _summarize_prediction(result)


vanguard_agent = Agent(
    name="VANGUARD Mars Assistant",
    instructions=INSTRUCTIONS,
    model=openai_model,
    output_type=AgentReply,
    tools=[
        find_best_landing_region,
        list_mars_sites,
        lookup_mars_site,
        get_vanguard_documentation,
        get_scoring_weights,
        set_scoring_weights,
        compare_landing_sites,
        focus_mars_site,
        focus_mars_coordinates,
        set_globe_surface_layer,
        analyze_landing_site,
    ],
)


async def run_agent_turn(
    user_message: str,
    scoring_weights: dict | None = None,
) -> dict[str, Any]:
    """Single user message → assistant reply text and optional UI actions for the globe."""
    actions: list[dict[str, Any]] = []
    token = _ui_actions.set(actions)
    t0 = time.perf_counter()
    _trace("turn start", user_preview=user_message[:160])
    try:
        from backend.scoring import parse_scoring_weights, scoring_weights_as_percent
    except ImportError:
        from scoring import parse_scoring_weights, scoring_weights_as_percent

    initial_sw = (
        parse_scoring_weights(scoring_weights) if scoring_weights is not None else None
    )
    sw_token = _scoring_weights.set(initial_sw)
    if initial_sw is not None:
        _trace("scoring_weights from client", weights=scoring_weights_as_percent(initial_sw))
    try:
        result = await Runner.run(vanguard_agent, user_message)
        parsed = _parse_agent_reply(result, vanguard_agent)
        reply = parsed.summary
        _trace_run_result(user_message, result, reply, actions)
        _trace("turn elapsed_s", seconds=round(time.perf_counter() - t0, 2))
        return {
            "reply": reply,
            "structured": parsed.model_dump(),
            "ui_actions": list(actions),
        }
    except Exception as exc:
        _trace("turn failed", error=str(exc), seconds=round(time.perf_counter() - t0, 2))
        raise
    finally:
        _ui_actions.reset(token)
        _scoring_weights.reset(sw_token)


async def _chat_repl() -> None:
    print("VANGUARD Mars Assistant (empty line or Ctrl+C to exit)\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            break
        out = await run_agent_turn(user)
        reply = out["reply"]
        if out.get("structured"):
            print(f"\n[structured: {json.dumps(out['structured'], indent=2)}]")
        if out.get("ui_actions"):
            print(f"\n[UI actions: {json.dumps(out['ui_actions'])}]")
        print(f"\nAssistant: {reply}\n")


def main() -> None:
    asyncio.run(_chat_repl())


if __name__ == "__main__":
    main()
