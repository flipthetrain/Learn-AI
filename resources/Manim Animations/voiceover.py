"""
voiceover.py — gTTS voiceover utility for Manim animations
===========================================================
Generates MP3 clips from narration text using Google Text-to-Speech (free).
Clips are cached on disk — only regenerated when the text changes.

Usage in an animation script:
    from voiceover import generate_voiceovers
    from pathlib import Path

    NARRATION = {
        "intro":   "Welcome to this animation about transformers.",
        "section": "The key insight is that attention is all you need.",
    }

    class MyScene(Scene):
        def construct(self):
            vo = generate_voiceovers(NARRATION, Path(__file__).stem)

            self.add_sound(vo.path("intro"))
            self.play(Write(title), run_time=2.0)
            self.wait(vo.wait("intro", used=2.0))

            self.add_sound(vo.path("section"))
            self.play(FadeIn(content), run_time=1.5)
            self.wait(vo.wait("section", used=1.5))
"""

import hashlib
import json
import sys
from pathlib import Path

# ── Optional dependencies ──────────────────────────────────────────────────────

try:
    from gtts import gTTS
    _GTTS = True
except ImportError:
    _GTTS = False

try:
    from mutagen.mp3 import MP3 as _MutagenMP3
    _MUTAGEN = True
except ImportError:
    _MUTAGEN = False

# Fallback speaking rate used when mutagen is unavailable
_WORDS_PER_SECOND = 2.6


# ── Voiceover result object ───────────────────────────────────────────────────

class Voiceover:
    """Holds paths and durations for a set of named audio clips."""

    def __init__(self, clips: dict):
        # clips: {name: {"path": str, "duration": float}}
        self._clips = clips

    def path(self, name: str) -> str:
        """Absolute path to the MP3 file for this segment."""
        return self._clips[name]["path"]

    def duration(self, name: str) -> float:
        """Duration of the clip in seconds."""
        return self._clips[name]["duration"]

    def wait(self, name: str, used: float = 0.0) -> float:
        """
        How long to self.wait() after `used` seconds of animation have played,
        so the next segment starts only after this one finishes speaking.
        Returns max(0, duration - used) — never negative.
        """
        return max(0.0, self.duration(name) - used)

    def exists(self, name: str) -> bool:
        """Return True only if the MP3 file was actually written to disk.

        Use this to guard self.add_sound() calls so rendering works even when
        gtts is not installed:

            if vo.exists("intro"):
                self.add_sound(vo.path("intro"))
        """
        return Path(self._clips[name]["path"]).exists()

    def __repr__(self):
        lines = [f"Voiceover({len(self._clips)} clips):"]
        for name, data in self._clips.items():
            lines.append(f"  {name:20s}  {data['duration']:.1f}s  {data['path']}")
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _estimate_duration(text: str) -> float:
    """Rough duration estimate based on word count (used when mutagen unavailable)."""
    return max(1.0, len(text.split()) / _WORDS_PER_SECOND)


def _mp3_duration(path: Path) -> float:
    if _MUTAGEN:
        try:
            return _MutagenMP3(str(path)).info.length
        except Exception:
            pass
    return _estimate_duration(path.stem.replace("_", " "))


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def generate_voiceovers(
    scripts: dict,
    scene_name: str | Path = "voiceover",
    lang: str = "en",
    slow: bool = False,
) -> Voiceover:
    """
    Generate (or load cached) MP3 voiceover clips for a Manim scene.

    Args:
        scripts:    {segment_name: narration_text} ordered dict.
        scene_name: Name of the scene / script stem. Clips are saved to
                    `<scene_name>_voiceover/` next to this module.
        lang:       BCP-47 language code passed to gTTS (default 'en').
        slow:       Use gTTS slow mode (default False).

    Returns:
        Voiceover object. Call .path(name), .duration(name), .wait(name, used).

    Notes:
        - Requires `pip install gtts` for generation.
        - Requires `pip install mutagen` for accurate durations.
        - Falls back to word-count estimate if mutagen is unavailable.
        - Clips are only regenerated when the text changes (MD5 cache).
    """
    if not _GTTS:
        print(
            "[voiceover] WARNING: gtts not installed. "
            "Run: pip install gtts\n"
            "[voiceover] Returning estimated durations — no audio will play."
        )

    # Output directory: <this_file_dir>/<scene_name>_voiceover/
    base_dir = Path(__file__).parent
    out_dir  = base_dir / f"{Path(scene_name).stem}_voiceover"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persistent cache: maps segment name → MD5 of last generated text
    cache_file = out_dir / ".cache.json"
    cache: dict = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    clips: dict = {}

    for name, text in scripts.items():
        mp3 = out_dir / f"{name}.mp3"
        current_hash = _text_hash(text)

        needs_regen = not mp3.exists() or cache.get(name) != current_hash

        if needs_regen:
            if _GTTS:
                print(f"  [voiceover] generating {name}.mp3 ...")
                try:
                    gTTS(text=text, lang=lang, slow=slow).save(str(mp3))
                    cache[name] = current_hash
                except Exception as exc:
                    print(f"  [voiceover] ERROR generating {name}.mp3: {exc}")
                    clips[name] = {"path": str(mp3), "duration": _estimate_duration(text)}
                    continue
            else:
                # No gTTS — store a placeholder duration, no file written
                clips[name] = {"path": str(mp3), "duration": _estimate_duration(text)}
                continue

        duration = _mp3_duration(mp3) if mp3.exists() else _estimate_duration(text)
        clips[name] = {"path": str(mp3), "duration": duration}

    # Persist updated cache
    cache_file.write_text(json.dumps(cache, indent=2))

    vo = Voiceover(clips)
    print(vo)
    return vo


# ── CLI: regenerate all clips for a single NARRATION dict ────────────────────

if __name__ == "__main__":
    import importlib.util, argparse

    parser = argparse.ArgumentParser(
        description="Pre-generate voiceover clips for a Manim animation script."
    )
    parser.add_argument("script", help="Path to the animation .py file")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--slow", action="store_true", help="Use gTTS slow mode")
    args = parser.parse_args()

    spec   = importlib.util.spec_from_file_location("anim", args.script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "NARRATION"):
        print(f"ERROR: {args.script} has no NARRATION dict.")
        sys.exit(1)

    vo = generate_voiceovers(
        module.NARRATION,
        scene_name=Path(args.script).stem,
        lang=args.lang,
        slow=args.slow,
    )
    print("\nDone.")
