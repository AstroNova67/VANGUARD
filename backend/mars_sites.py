"""Famous Mars locations (aligned with frontend/3d_globe/index.js MARS_FAMOUS_LOCATIONS)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarsSite:
    id: str
    name: str
    lat: float
    lon: float


MARS_FAMOUS_SITES: tuple[MarsSite, ...] = (
    MarsSite("jezero", "Jezero crater (Perseverance)", 18.4447, 77.4508),
    MarsSite("gale", "Gale crater (Curiosity)", -5.5892, 137.4417),
    MarsSite("meridiani", "Meridiani Planum (Opportunity)", -1.9462, -5.5266),
    MarsSite("gusev", "Gusev crater (Spirit)", -14.5689, 175.4726),
    MarsSite("viking1", "Viking 1 (Chryse Planitia)", 22.4872, -47.9424),
    MarsSite("viking2", "Viking 2 (Utopia Planitia)", 47.967, 134.991),
    MarsSite("insight", "Elysium Planitia (InSight)", 4.5024, 135.6234),
    MarsSite("pathfinder", "Ares Vallis (Pathfinder)", 19.3278, -33.5441),
    MarsSite("phoenix", "Green Valley (Phoenix)", 68.2188, -125.7497),
    MarsSite("olympus", "Olympus Mons (caldera)", 18.65, -133.8),
    MarsSite("valles", "Valles Marineris (Coprates vicinity)", -14.5, -59.0),
    MarsSite("hellas", "Hellas Planitia (basin center)", -42.0, 70.0),
    MarsSite("noctis", "Noctis Labyrinthus", -7.0, -97.0),
    MarsSite("ascraeus", "Ascraeus Mons", 11.2, -104.1),
)


def find_mars_site(query: str) -> MarsSite | None:
    """Case-insensitive match on id or name (substring)."""
    q = query.strip().lower()
    if not q:
        return None
    for site in MARS_FAMOUS_SITES:
        if q == site.id or q in site.name.lower():
            return site
    for site in MARS_FAMOUS_SITES:
        if q in site.id:
            return site
    return None


def format_sites_list() -> str:
    lines = []
    for s in MARS_FAMOUS_SITES:
        lines.append(f"- {s.name} (id={s.id}): lat={s.lat}°N, lon={s.lon}°E")
    return "\n".join(lines)
