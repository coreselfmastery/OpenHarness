"""BEAST — the one tier (OpenVideoHarness playbook/01-pipeline.md).

The film rendered ONCE at ship grade: 720p-class, NVENC p5/cq21 when a card
answers. These tests pin the OpenHarness side of the contract so the tier
cannot silently drift: it must be a valid tier everywhere a tier is checked,
resolve to 1280x720, and substitute into engine command templates via {tier}
so render.mjs --tier beast / blender --profile beast receive it verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

from openharness.services.render_pipeline import (
    _FFMPEG_INPUT_ARGS,
    _MANIM_QUALITY,
    TIERS,
    TIER_ORDER,
    build_render_command,
    load_engine_templates,
)


def test_beast_is_a_valid_tier():
    assert "beast" in TIERS
    assert TIERS[-1] == "beast"  # top of the ladder, meant to be used alone
    assert TIER_ORDER["beast"] > TIER_ORDER["final"]


def test_beast_resolves_720p_class():
    assert "1280x720" in _FFMPEG_INPUT_ARGS["beast"]
    assert _MANIM_QUALITY["beast"] == "m"  # manim's 720p quality flag


def test_beast_flows_through_the_tier_placeholder(tmp_path: Path):
    page_bridge = [
        "bin/vh", "render", "{scene}",
        "--tier", "{tier}", "--out", "{output}",
    ]
    cmd = build_render_command(
        engine="page",
        scene_path=Path("scenes/hero.html"),
        output_path=tmp_path / "hero.beast.mp4",
        tier="beast",
        width=1280,
        height=720,
        fps=30,
        codec_args="",
        templates={"page": page_bridge},
        output_dir=tmp_path,
    )
    assert cmd[cmd.index("--tier") + 1] == "beast"


def test_beast_engine_json_overrides(tmp_path: Path):
    (tmp_path / "render_engines.json").write_text(
        json.dumps(
            {
                "page": [
                    "bin/vh", "render", "{scene}",
                    "--tier", "{tier}", "--out", "{output}",
                ]
            }
        ),
        encoding="utf-8",
    )
    templates = load_engine_templates(tmp_path)
    cmd = build_render_command(
        engine="page",
        scene_path=Path("film.html"),
        output_path=tmp_path / "film.beast.mp4",
        tier="beast",
        width=1280,
        height=720,
        fps=30,
        codec_args="",
        templates=templates,
        output_dir=tmp_path,
    )
    assert cmd[cmd.index("--tier") + 1] == "beast"
