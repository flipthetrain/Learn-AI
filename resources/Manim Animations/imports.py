# This script checks and installs all dependencies required by the Manim animations.
# Run this before rendering any animation for the first time.
#
# NOTE: manim also requires system-level dependencies that pip cannot install:
#   - FFmpeg  : video encoding
#   - LaTeX   : math equation rendering (MathTex scenes)
#
# See readme.md for platform-specific install instructions (macOS / Ubuntu / Windows).

import sys
import os
import subprocess


# ── Sanity checks before doing anything ───────────────────────────────────────

in_venv = sys.prefix != sys.base_prefix
if not in_venv:
    print(
        "WARNING: no virtual environment is active.\n"
        "It is strongly recommended to use a venv:\n\n"
        "    python3 -m venv .venv\n"
        "    source .venv/bin/activate   # Linux/macOS\n"
        "    .venv\\Scripts\\activate      # Windows\n"
        "    python imports.py\n\n"
        "Continuing anyway — packages will be installed to the system Python.\n"
    )


# ── Install and verify system libraries ───────────────────────────────────────

def ensure_system_deps():
    """Install missing C libraries and CLI tools via apt (Linux) or report on other platforms."""
    import shutil
    import platform

    # pkg-config checks for C libraries
    c_libs = {
        "pangocairo": "libpango1.0-dev",
        "cairo":      "libcairo2-dev",
    }
    # CLI tools
    cli_tools = {
        "pkg-config": "pkg-config",
        "ffmpeg":     "ffmpeg",
    }

    missing = []

    for lib, apt_pkg in c_libs.items():
        r = subprocess.run(["pkg-config", "--exists", lib], capture_output=True)
        if r.returncode != 0:
            missing.append(apt_pkg)

    for cmd, apt_pkg in cli_tools.items():
        if not shutil.which(cmd):
            missing.append(apt_pkg)

    if not missing:
        print("System dependencies: OK")
        return

    system = platform.system()

    if system == "Linux" and shutil.which("apt"):
        pkgs = " ".join(dict.fromkeys(missing))   # deduplicate, preserve order
        print(f"Installing system packages: {pkgs}")
        result = subprocess.run(
            ["sudo", "apt", "install", "-y", "python3-dev"] + list(dict.fromkeys(missing)),
            text=True,
        )
        if result.returncode != 0:
            print(
                "ERROR: apt install failed. Try manually:\n\n"
                f"    sudo apt install -y python3-dev {pkgs}\n"
            )
            sys.exit(1)
        print("System dependencies: installed")

    elif system == "Darwin" and shutil.which("brew"):
        # macOS: ffmpeg via brew; pango/cairo come with cairo cask
        brew_pkgs = {"ffmpeg": "ffmpeg", "libpango1.0-dev": "pango", "libcairo2-dev": "cairo"}
        to_brew = list(dict.fromkeys(brew_pkgs.get(p, p) for p in missing))
        print(f"Installing via Homebrew: {' '.join(to_brew)}")
        subprocess.check_call(["brew", "install"] + to_brew)

    else:
        pkgs = " ".join(dict.fromkeys(missing))
        print(
            f"ERROR: missing system packages: {pkgs}\n"
            "Install them manually then re-run this script.\n"
            "  Ubuntu/Debian : sudo apt install -y python3-dev " + pkgs + "\n"
            "  macOS         : brew install ffmpeg pango cairo\n"
            "  Windows       : see readme.md\n"
        )
        sys.exit(1)


ensure_system_deps()


def install(package):
    """Install a package into the current Python environment."""
    cmd = [sys.executable, "-m", "pip", "install", package]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    # Debian/Ubuntu externally-managed-environment: retry with the override flag
    if "externally-managed-environment" in result.stderr:
        print(f"  (retrying with --break-system-packages)")
        subprocess.check_call(cmd + ["--break-system-packages"])
    else:
        print(result.stderr.strip())
        raise subprocess.CalledProcessError(result.returncode, cmd)


required_packages = [
    "manim",      # Manim Community Edition — animation engine
    "numpy",      # Numerical computing (used internally by manim)
    "Pillow",     # Image processing (used internally by manim)
    "scipy",      # Scientific computing (used internally by manim)
    "colour",     # Colour math (used internally by manim)
    "tqdm",       # Progress bars
    "requests",   # HTTP (manim asset downloads)
    "rich",       # Terminal formatting (manim CLI output)
    "click",      # CLI framework (manim CLI)
    "cloup",      # CLI helpers (manim CLI)
    "pycairo",    # 2D vector graphics renderer
    "manimpango", # Text / font rendering via Pango
    "pygments",   # Syntax highlighting (Code mobject)
    "isosurfaces", # 3D surface rendering (ThreeDScene)
    "moderngl",   # OpenGL rendering backend
    "screeninfo",  # Display resolution detection
    "gtts",       # Google Text-to-Speech — voiceover generation
    "mutagen",    # MP3 duration reading (voiceover timing)
]

IMPORT_NAME_OVERRIDES = {
    "Pillow":      "PIL",
    "pycairo":     "cairo",
    "isosurfaces": "isosurfaces",
}

for pkg in required_packages:
    import_name = IMPORT_NAME_OVERRIDES.get(pkg, pkg.replace("-", "_"))
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)
    except Exception:
        pass   # some packages (e.g. moderngl) may raise on headless systems

# Verify manim itself is importable
try:
    import manim
    print(f"manim version : {manim.__version__}")
except ImportError:
    print(
        "\nERROR: manim is not importable even after install.\n"
        "This usually means ffmpeg or LaTeX is missing from your system PATH.\n"
        "See readme.md for platform-specific setup instructions."
    )
    sys.exit(1)

# Check ffmpeg is available on PATH
import shutil
if shutil.which("ffmpeg"):
    print("ffmpeg        : found")
else:
    print("WARNING: ffmpeg not found on PATH — animations cannot be rendered without it.")
    print("  macOS  : brew install ffmpeg")
    print("  Ubuntu : sudo apt install ffmpeg")
    print("  Windows: https://ffmpeg.org/download.html")

# Check a LaTeX executable is available (pdflatex or xelatex)
latex_found = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("lualatex")
if latex_found:
    print(f"LaTeX         : found ({latex_found})")
else:
    print("WARNING: no LaTeX executable found — MathTex scenes will fail.")
    print("  macOS  : brew install --cask mactex-no-gui")
    print("  Ubuntu : sudo apt install texlive texlive-latex-extra")
    print("  Windows: https://miktex.org/download")

print("\nAll checks complete. To render an animation:")
print("  manim -ql <script>.py <ClassName>")
print("  manim -qh <script>.py <ClassName>   # high quality")
