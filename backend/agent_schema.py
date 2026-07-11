"""
Pydantic schemas for Mars assistant structured output.

Used with OpenAI Agents SDK ``Agent(output_type=...)``:
https://openai.github.io/openai-agents-python/agents/#output-types

When ``output_type`` is set, the model uses structured outputs and
``RunResult.final_output`` is an instance of that type (not plain text).
See: https://openai.github.io/openai-agents-python/results/#final-output
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PropertyContribution(BaseModel):
    """One scored property's share of the landing suitability total."""

    model_config = ConfigDict(extra="forbid")

    property_name: str = Field(
        description="slope | dust | surface_temp | thermal_inertia | water"
    )
    contribution_percent: float = Field(
        ge=0,
        le=100,
        description="Weighted contribution to landing score (0–100).",
    )


class ScoringWeightsPercent(BaseModel):
    """Fixed keys for strict JSON schema (no open dict)."""

    model_config = ConfigDict(extra="forbid")

    slope: float | None = Field(default=None, ge=0, le=100)
    dust: float | None = Field(default=None, ge=0, le=100)
    surface_temp: float | None = Field(default=None, ge=0, le=100)
    thermal_inertia: float | None = Field(default=None, ge=0, le=100)
    water: float | None = Field(default=None, ge=0, le=100)

    @classmethod
    def from_mapping(cls, raw: dict[str, float] | None) -> ScoringWeightsPercent | None:
        if not raw:
            return None
        return cls(
            slope=raw.get("slope"),
            dust=raw.get("dust"),
            surface_temp=raw.get("surface_temp"),
            thermal_inertia=raw.get("thermal_inertia"),
            water=raw.get("water"),
        )


class BestLandingSite(BaseModel):
    """Single recommended landing location (use for 'find a good region' requests)."""

    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(description="Degrees north (−90…90).")
    longitude: float = Field(description="Degrees east (−180…180).")
    landing_score_percent: float = Field(ge=0, le=100)
    interpretation: str = Field(
        description="Score band label, e.g. Excellent / Fair / Poor."
    )
    region_description: str | None = Field(
        default=None,
        description="Short plain label, e.g. 'Northern lowlands' (not a mission name unless user asked).",
    )
    top_contributions: list[PropertyContribution] = Field(
        default_factory=list,
        max_length=5,
        description="Largest score contributions from score_breakdown.",
    )


class AgentReply(BaseModel):
    """
    Final assistant message returned to the chat UI.
    Keep summary short; put coordinates and scores in best_site when applicable.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description=(
            "Plain text only: 1–3 concise sentences for the user. "
            "No HTML, no numbered lists, no markdown headings."
        )
    )
    intent: Literal[
        "explain",
        "best_region",
        "analyze_site",
        "compare",
        "navigate",
        "settings",
        "general",
    ] = Field(
        default="general",
        description="best_region when recommending a single optimal location from tools.",
    )
    best_site: BestLandingSite | None = Field(
        default=None,
        description="Required when intent is best_region; otherwise null.",
    )
    scoring_weights_percent: ScoringWeightsPercent | None = Field(
        default=None,
        description="Active scoring weights (percent) if relevant to the answer.",
    )
