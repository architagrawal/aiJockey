"""Preset → env + CLI args mapping for /generate endpoint.

Single source of truth for translating user-facing UI choices
(mode / vocals / style / arc / mix_mode) into the env vars and CLI args
that main.py + execute.py + master.py read.

Usage in server/api.py /generate handler:

    from preset import apply_preset, PRESET_SCHEMA

    env_overrides, cli_overrides = apply_preset(
        mode=mode,                    # 'mashup' | 'dj_set'
        vocals=vocals,                # 'on' | 'off' | 'dim'
        style=style,                  # PRESETS key or None
        arc=arc,                      # arc preset name or None (mode supplies default)
        mix_mode=mix_mode,            # 'tight' | 'balanced' | 'exploratory'
        advanced=advanced_dict,       # optional {knob: value} from advanced panel
    )
    subprocess_env = {**os.environ, **env_overrides}
    cli_args = base_cli + cli_overrides
"""
from __future__ import annotations

from typing import Any


# --- Schema for frontend (radio/dropdown options) ---------------------------

PRESET_SCHEMA = {
    "mode":     {"type": "radio",    "values": ["mashup", "dj_set"],
                 "default": "dj_set",
                 "labels": {"mashup": "Mashup (polished)",
                            "dj_set": "DJ Set (festival)"}},
    "vocals":   {"type": "radio",    "values": ["on", "off", "dim"],
                 "default": "on"},
    "style":    {"type": "dropdown", "values": ["", "festival_inferno",
                                                 "midnight_noir",
                                                 "neon_retrowave",
                                                 "east_meets_bass",
                                                 "bollywood_block_party"],
                 "default": ""},
    "arc":      {"type": "dropdown", "values": ["", "build", "peak",
                                                 "tomorrowland", "rollercoaster",
                                                 "descend", "flat_high",
                                                 "flat_low"],
                 "default": ""},
    "mix_mode": {"type": "radio",    "values": ["tight", "balanced", "exploratory"],
                 "default": "balanced"},
    "duration": {"type": "slider",   "min": 60, "max": 600, "default": 300},
}

# Advanced panel toggles — flat dict, all optional
ADVANCED_SCHEMA = {
    "lufs":                  {"type": "slider", "min": -14.0, "max": -7.0, "default": -9.0},
    "tape_sat":              {"type": "bool", "default": True},
    "tape_drive":            {"type": "slider", "min": 0.0, "max": 1.0, "default": 0.6},
    "mel_band":              {"type": "bool", "default": True},
    "dtw_align":             {"type": "bool", "default": True},
    "use_director":          {"type": "bool", "default": True},
    "tier_enforce":          {"type": "bool", "default": True},
    "max_minor_pct":         {"type": "slider", "min": 0.5, "max": 1.0, "default": 0.75},
    "force_drop":            {"type": "bool", "default": False},
    "auto_accents":          {"type": "bool", "default": False},
    "vocal_guard":           {"type": "bool", "default": True},
    "vocal_guard_thr":       {"type": "slider", "min": 0.2, "max": 0.5, "default": 0.30},
    "vocal_guard_thr_heavy": {"type": "slider", "min": 0.4, "max": 0.7, "default": 0.55},
    "vocal_end_snap":        {"type": "bool", "default": True},
    "vocal_safe_stretch":    {"type": "bool", "default": True},
    "vocal_ramp_beats":      {"type": "slider", "min": 2, "max": 8, "default": 4},
    "phrase_quantize":       {"type": "bool", "default": True},
    "stem_swap":             {"type": "bool", "default": True},
    "candidate_picker":      {"type": "bool", "default": True},
    "all_in_one":            {"type": "bool", "default": True},
    "audiobox_critic":       {"type": "bool", "default": True},
    "constitutional":        {"type": "bool", "default": True},
    "n_best":                {"type": "slider", "min": 1, "max": 8, "default": 1},
}


# --- Preset definitions -----------------------------------------------------

_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "mashup": {
        "env": {
            "AIJOCKEY_MASTER_TAPE_SAT": "0",
            "AIJOCKEY_DIRECTOR_TIER_ENFORCE": "1",
            "AIJOCKEY_DIRECTOR_MAX_MINOR_PCT": "0.85",
            "AIJOCKEY_VOCAL_GUARD_THR": "0.30",
            "AIJOCKEY_VOCAL_GUARD_THR_HEAVY": "0.45",
            "AIJOCKEY_VOCAL_END_SNAP": "1",
            "AIJOCKEY_VOCAL_RAMP_BEATS": "4",
        },
        "cli": {
            "arc": "build",
            "callbacks": 2,
            "reuse_cooldown": 5,
            "max_clips": 12,
        },
        "planner_overrides": {
            # Threaded via env reads inside planner.py if exposed; otherwise
            # passed as additional CLI args when caller threads them through.
            "min_segment_seconds": 28.0,
            "max_segment_seconds": 0.0,   # 0 = no cap
        },
    },
    "dj_set": {
        "env": {
            "AIJOCKEY_MASTER_TAPE_SAT": "1",
            "AIJOCKEY_MASTER_TAPE_DRIVE": "0.3",
            "AIJOCKEY_DIRECTOR_TIER_ENFORCE": "1",
            "AIJOCKEY_DIRECTOR_MAX_MINOR_PCT": "0.65",
            "AIJOCKEY_VOCAL_GUARD_THR": "0.40",
            "AIJOCKEY_VOCAL_GUARD_THR_HEAVY": "0.60",
            "AIJOCKEY_VOCAL_END_SNAP": "1",
            "AIJOCKEY_VOCAL_RAMP_BEATS": "4",
        },
        "cli": {
            "arc": "tomorrowland",
            "callbacks": 4,
            "reuse_cooldown": 1,
            "max_clips": 16,
        },
        "planner_overrides": {
            "min_segment_seconds": 18.0,
            "max_segment_seconds": 28.0,
        },
    },
}

_VOCAL_PRESETS: dict[str, dict[str, str]] = {
    "on":  {"AIJOCKEY_INSTRUMENTAL_ONLY": "0", "AIJOCKEY_STEM_SWAP": "1"},
    "off": {"AIJOCKEY_INSTRUMENTAL_ONLY": "1"},
    "dim": {"AIJOCKEY_INSTRUMENTAL_ONLY": "0", "AIJOCKEY_STEM_SWAP": "1",
            "AIJOCKEY_VOCAL_DIM_DB": "-6"},  # NEW knob — execute.py needs ~5 LOC
}

# CLAP-text + arc bias from semantic preset
STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "festival_inferno":     {"arc": "peak",          "prompt_suffix": " festival main stage euphoric drops"},
    "midnight_noir":        {"arc": "flat_low",      "prompt_suffix": " smoky lo-fi noir after-hours"},
    "neon_retrowave":       {"arc": "rollercoaster", "prompt_suffix": " 80s synthwave neon nostalgia"},
    "east_meets_bass":      {"arc": "rollercoaster", "prompt_suffix": " sitar tabla deep bass fusion"},
    "bollywood_block_party":{"arc": "build",         "prompt_suffix": " bollywood club punjabi dancefloor"},
}


# --- Advanced knob → env mapping --------------------------------------------

_ADV_TO_ENV = {
    "tape_sat":              ("AIJOCKEY_MASTER_TAPE_SAT", lambda v: "1" if v else "0"),
    "tape_drive":            ("AIJOCKEY_MASTER_TAPE_DRIVE", lambda v: f"{float(v):.2f}"),
    "mel_band":              ("AIJOCKEY_MEL_BAND_ROFORMER", lambda v: "1" if v else "0"),
    "dtw_align":             ("AIJOCKEY_DTW_ALIGN", lambda v: "1" if v else "0"),
    "use_director":          ("AIJOCKEY_USE_DIRECTOR_LLM", lambda v: "1" if v else "0"),
    "tier_enforce":          ("AIJOCKEY_DIRECTOR_TIER_ENFORCE", lambda v: "1" if v else "0"),
    "max_minor_pct":         ("AIJOCKEY_DIRECTOR_MAX_MINOR_PCT", lambda v: f"{float(v):.2f}"),
    "force_drop":            ("AIJOCKEY_DIRECTOR_FORCE_DROP", lambda v: "1" if v else "0"),
    "auto_accents":          ("AIJOCKEY_DIRECTOR_AUTO_ACCENT", lambda v: "1" if v else "0"),
    "vocal_guard":           ("AIJOCKEY_VOCAL_GUARD", lambda v: "1" if v else "0"),
    "vocal_guard_thr":       ("AIJOCKEY_VOCAL_GUARD_THR", lambda v: f"{float(v):.2f}"),
    "vocal_guard_thr_heavy": ("AIJOCKEY_VOCAL_GUARD_THR_HEAVY", lambda v: f"{float(v):.2f}"),
    "vocal_end_snap":        ("AIJOCKEY_VOCAL_END_SNAP", lambda v: "1" if v else "0"),
    "vocal_safe_stretch":    ("AIJOCKEY_VOCAL_SAFE_STRETCH", lambda v: "1" if v else "0"),
    "vocal_ramp_beats":      ("AIJOCKEY_VOCAL_RAMP_BEATS", lambda v: str(int(v))),
    "phrase_quantize":       ("AIJOCKEY_PHRASE_QUANTIZE", lambda v: "1" if v else "0"),
    "stem_swap":             ("AIJOCKEY_STEM_SWAP", lambda v: "1" if v else "0"),
    "candidate_picker":      ("AIJOCKEY_CANDIDATE_PICKER", lambda v: "1" if v else "0"),
    "all_in_one":            ("AIJOCKEY_ALL_IN_ONE", lambda v: "1" if v else "0"),
    "audiobox_critic":       ("AIJOCKEY_AUDIOBOX_AESTHETICS", lambda v: "1" if v else "0"),
    "constitutional":        ("AIJOCKEY_CONSTITUTIONAL", lambda v: "1" if v else "0"),
}

_ADV_TO_CLI = {
    "lufs":   ("lufs", float),
    "n_best": ("n_best", int),
}


# --- Public API -------------------------------------------------------------

def apply_preset(
    mode: str = "dj_set",
    vocals: str = "on",
    style: str | None = None,
    arc: str | None = None,
    mix_mode: str = "balanced",
    advanced: dict[str, Any] | None = None,
    base_prompt: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Compose user UI choices into env-var dict + CLI-arg dict.

    Returns:
        (env_overrides, cli_overrides)
        env_overrides: merge into os.environ for subprocess.run(env=...)
        cli_overrides: dict — caller maps to actual CLI flags
            ('arc', 'callbacks', 'reuse_cooldown', 'max_clips',
             'mix_mode', 'prompt', 'lufs', 'n_best',
             'min_segment_seconds', 'max_segment_seconds')
    """
    if mode not in _MODE_PRESETS:
        raise ValueError(f"unknown mode: {mode}; must be {list(_MODE_PRESETS)}")
    if vocals not in _VOCAL_PRESETS:
        raise ValueError(f"unknown vocals: {vocals}; must be {list(_VOCAL_PRESETS)}")

    env: dict[str, str] = {}
    cli: dict[str, Any] = {"mix_mode": mix_mode}

    # 1. mode preset (foundation)
    mp = _MODE_PRESETS[mode]
    env.update(mp["env"])
    cli.update(mp["cli"])
    cli.update(mp.get("planner_overrides", {}))

    # 2. vocals overlay
    env.update(_VOCAL_PRESETS[vocals])

    # 3. style preset (CLAP-text + arc hint)
    prompt = base_prompt or ""
    if style and style in STYLE_PRESETS:
        sp = STYLE_PRESETS[style]
        prompt = (prompt + sp["prompt_suffix"]).strip()
        # style arc only overrides if arc not explicitly set
        if not arc:
            cli["arc"] = sp["arc"]
    if prompt:
        cli["prompt"] = prompt

    # 4. explicit arc override (highest priority)
    if arc:
        cli["arc"] = arc

    # 5. advanced knob overrides — walk dict, dispatch to env or cli
    if advanced:
        for k, v in advanced.items():
            if k in _ADV_TO_ENV:
                env_key, conv = _ADV_TO_ENV[k]
                env[env_key] = conv(v)
            elif k in _ADV_TO_CLI:
                cli_key, conv = _ADV_TO_CLI[k]
                cli[cli_key] = conv(v)

    return env, cli


def compose_cli_args(cli_overrides: dict[str, Any]) -> list[str]:
    """Convert cli_overrides dict into list of CLI args for main.py plan call.

    Skips planner_overrides keys (caller threads those through PlannerConfig).
    """
    args: list[str] = []
    flag_map = {
        "arc": "--arc", "callbacks": "--callbacks",
        "reuse_cooldown": "--reuse_cooldown",
        "max_clips": "--max_clips", "prompt": "--prompt",
        "n_best": "--n_best",
    }
    for k, v in cli_overrides.items():
        if k in flag_map:
            args.extend([flag_map[k], str(v)])
    return args


# --- Self-test --------------------------------------------------------------

if __name__ == "__main__":
    import json
    for mode in ("mashup", "dj_set"):
        for vocals in ("on", "off"):
            env, cli = apply_preset(mode=mode, vocals=vocals,
                                     style="festival_inferno",
                                     base_prompt="user prompt here",
                                     advanced={"tape_drive": 0.4,
                                               "lufs": -8.0})
            print(f"\n=== mode={mode} vocals={vocals} ===")
            print("ENV:", json.dumps(env, indent=2))
            print("CLI:", json.dumps(cli, indent=2, default=str))
            print("CLI ARGS:", compose_cli_args(cli))
