"""Kokoro-82M TTS wrapper for short DJ-tag voiceovers.

Apache-2.0, 82M params, 24 kHz mono, ~20x realtime. 8 langs (en-US/en-GB/
ja/zh/es/fr/hi/it). Used here for ≤6-word DJ shouts like "drop it",
"welcome to the floor", language-matched to the mix's style preset.

External dep: `espeak-ng` (apt) + `kokoro` pip pkg.

Env knobs:
    AIJOCKEY_KOKORO_ENABLE   0|1   default 0
    AIJOCKEY_KOKORO_DEVICE   str   default 'cuda' if available
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPES: dict[str, object] = {}   # lang_code -> KPipeline
_LOAD_FAILED = False


# Map our `style` presets to (Kokoro lang_code, default voice).
# Voices verified per VOICES.md in hexgrad/Kokoro-82M; pick warm/energetic
# voices appropriate for DJ tags.
STYLE_TO_LOCALE: dict[str, tuple[str, str]] = {
    "festival_inferno":      ("a", "af_heart"),
    "midnight_noir":         ("b", "bf_alice"),
    "neon_retrowave":        ("a", "am_michael"),
    "east_meets_bass":       ("j", "jf_alpha"),
    "bollywood_block_party": ("h", "hf_alpha"),
    # Fallback if no style or unknown:
    "default":               ("a", "af_heart"),
}


# Default phrasebook per locale. Keep ≤6 words for quality. Hindi/es/fr
# entries kept simple. Caller can override via `phrases` arg.
DEFAULT_PHRASES: dict[str, list[str]] = {
    "a": ["welcome to the floor", "let's go", "drop it now",
           "feel the rhythm", "hands in the air", "make some noise"],
    "b": ["welcome to the floor", "feel the rhythm",
           "let the bass drop", "all night long"],
    "j": ["フロアへようこそ", "盛り上がろう", "落とせ"],
    "z": ["欢迎来到舞池", "把手举起来", "感受节奏"],
    "e": ["bienvenidos a la pista", "que suene fuerte", "vamos"],
    "f": ["bienvenue sur le dancefloor", "lâchez tout", "on y va"],
    "h": ["dance floor par swagat hai", "haath uthao", "drop karo"],
    "i": ["benvenuti in pista", "alzate le mani", "andiamo"],
}


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_KOKORO_ENABLE", "0") != "1":
        return False
    return not _LOAD_FAILED


def _get_pipeline(lang_code: str):
    """Lazy per-language KPipeline cache. Returns pipeline or None."""
    global _LOAD_FAILED
    if _LOAD_FAILED:
        return None
    with _LOCK:
        if lang_code in _PIPES:
            return _PIPES[lang_code]
        try:
            from kokoro import KPipeline  # type: ignore
        except Exception as e:
            print(f"[kokoro] import failed ({e}); install with `pip install kokoro`")
            _LOAD_FAILED = True
            return None
        try:
            pipe = KPipeline(lang_code=lang_code)
            _PIPES[lang_code] = pipe
            print(f"[kokoro] loaded pipeline for lang_code={lang_code}")
            return pipe
        except Exception as e:
            print(f"[kokoro] init failed for lang={lang_code}: {e}")
            return None


def render_tag(text: str, *, lang_code: str = "a",
                voice: str = "af_heart",
                out_path: str | Path,
                target_sr: int = 44100) -> str | None:
    """Synthesize a short tag and write to out_path as WAV at target_sr.

    Kokoro outputs 24 kHz mono. We resample to target_sr (typically the
    AiJockey mix SR = 44100) so downstream overlay needs no extra step.
    Returns out_path on success, None on failure.
    """
    if not enabled():
        return None
    pipe = _get_pipeline(lang_code)
    if pipe is None:
        return None
    try:
        import numpy as np
        import soundfile as sf
        # KPipeline yields (graphemes, phonemes, audio np.ndarray @ 24k).
        chunks = []
        for _, _, audio in pipe(text, voice=voice):
            if audio is None:
                continue
            chunks.append(np.asarray(audio, dtype="float32"))
        if not chunks:
            return None
        wav = np.concatenate(chunks)
        if target_sr and target_sr != 24000:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=24000,
                                        target_sr=target_sr).astype("float32")
            except Exception:
                target_sr = 24000   # fallback: write at native rate
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, wav, target_sr)
        return out_path
    except Exception as e:
        print(f"[kokoro] render failed for text={text!r}: {e}")
        return None


def locale_for_style(style: str | None) -> tuple[str, str]:
    """Return (lang_code, voice) tuple for AiJockey style preset."""
    if not style:
        return STYLE_TO_LOCALE["default"]
    return STYLE_TO_LOCALE.get(style.lower(), STYLE_TO_LOCALE["default"])


def render_phrasebook(out_dir: str | Path,
                       phrases: dict[str, list[str]] | None = None,
                       voices_per_lang: dict[str, list[str]] | None = None,
                       target_sr: int = 44100) -> list[dict]:
    """Pre-render a phrasebook of tags. Returns list of {lang, voice,
    text, path}.

    Use this at cache-build time to make tag overlay zero-latency at
    mix-render time.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    phrases = phrases or DEFAULT_PHRASES
    # Default voice per lang from STYLE_TO_LOCALE.
    default_voice_per_lang = {lc: v for (lc, v) in STYLE_TO_LOCALE.values()}
    manifest: list[dict] = []
    for lang_code, texts in phrases.items():
        voices = (voices_per_lang or {}).get(lang_code) or \
                 [default_voice_per_lang.get(lang_code, "af_heart")]
        for voice in voices:
            for txt in texts:
                slug = (f"{lang_code}_{voice}_"
                        + "".join(c if c.isalnum() else "_" for c in txt)[:40])
                target = out / f"{slug}.wav"
                if target.exists():
                    manifest.append({"lang_code": lang_code, "voice": voice,
                                      "text": txt, "path": str(target),
                                      "cached": True})
                    continue
                res = render_tag(txt, lang_code=lang_code, voice=voice,
                                  out_path=target, target_sr=target_sr)
                if res:
                    manifest.append({"lang_code": lang_code, "voice": voice,
                                      "text": txt, "path": res,
                                      "cached": False})
    import json
    (out / "phrasebook.json").write_text(json.dumps(manifest, indent=2))
    return manifest
