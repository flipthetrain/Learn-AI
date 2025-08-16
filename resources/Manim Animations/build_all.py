"""
build_all.py — Render all Manim animations in this folder.

Each script is rendered into a subfolder named after the script:
  lora_rank_decomposition.py  →  lora_rank_decomposition/videos/...
  rag_pipeline.py             →  rag_pipeline/videos/...
  ...

Usage:
    python build_all.py           # 480p preview (fast)
    python build_all.py --medium  # 720p
    python build_all.py --hq      # 1080p 60fps
    python build_all.py --4k      # 2160p 60fps
"""

import ast
import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Scripts to skip ────────────────────────────────────────────────────────────
SKIP = {"build_all.py", "imports.py", "setup.bat"}

# ── Python executable: prefer local .venv if present ─────────────────────────
def _find_python() -> str:
    """Return the Python executable to use for rendering.

    Priority:
      1. <this_dir>/.venv/bin/python  (Linux/macOS)
      2. <this_dir>/.venv/Scripts/python.exe  (Windows)
      3. sys.executable (current interpreter)
    """
    here = Path(__file__).parent
    for candidate in [
        here / ".venv" / "bin" / "python",
        here / ".venv" / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable

PYTHON = _find_python()


def find_scene_classes(path: Path) -> list[str]:
    """Return names of all classes that inherit from a Manim Scene base class."""
    SCENE_BASES = {"Scene", "ThreeDScene", "MovingCameraScene", "ZoomedScene"}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    scenes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = {
                b.id if isinstance(b, ast.Name) else
                b.attr if isinstance(b, ast.Attribute) else ""
                for b in node.bases
            }
            if bases & SCENE_BASES:
                scenes.append(node.name)
    return scenes


def render(script: Path, scene: str, quality_flag: str) -> bool:
    """Run manim for one scene. Returns True on success."""
    output_dir = script.parent / script.stem
    cmd = [
        PYTHON, "-m", "manim",
        quality_flag,
        "--media_dir", str(output_dir),
        str(script),
        scene,
    ]
    print(f"\n  → {script.name}::{scene}")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Render all Manim animations.")
    quality = parser.add_mutually_exclusive_group()
    quality.add_argument("--medium", action="store_true", help="720p 30fps")
    quality.add_argument("--hq",     action="store_true", help="1080p 60fps")
    quality.add_argument("--4k",     dest="fourk", action="store_true", help="2160p 60fps")
    args = parser.parse_args()

    if args.fourk:
        quality_flag = "-qk"
    elif args.hq:
        quality_flag = "-qh"
    elif args.medium:
        quality_flag = "-qm"
    else:
        quality_flag = "-ql"   # default: 480p fast preview

    here = Path(__file__).parent
    scripts = sorted(p for p in here.glob("*.py") if p.name not in SKIP)

    if not scripts:
        print("No animation scripts found.")
        sys.exit(0)

    quality_names = {"-ql": "480p (preview)", "-qm": "720p", "-qh": "1080p", "-qk": "2160p"}
    print(f"Rendering {len(scripts)} script(s) at {quality_names[quality_flag]}")
    print("=" * 60)

    passed, failed = [], []
    t0 = time.time()

    for script in scripts:
        scenes = find_scene_classes(script)
        if not scenes:
            print(f"\n  [skip] {script.name} — no Scene classes found")
            continue

        print(f"\n[{script.stem}]  ({len(scenes)} scene(s))")
        for scene in scenes:
            ok = render(script, scene, quality_flag)
            (passed if ok else failed).append(f"{script.name}::{scene}")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s — {len(passed)} succeeded, {len(failed)} failed")

    if passed:
        print("\nRendered:")
        for name in passed:
            stem = name.split("::")[0].replace(".py", "")
            print(f"  {stem}/videos/")

    if failed:
        print("\nFailed:")
        for name in failed:
            print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
